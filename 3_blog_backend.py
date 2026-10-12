from __future__ import annotations
import operator
from typing import TypedDict, List, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import ollama
from pathlib import Path
from model import MODEL
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from prompts.prompts_3 import WORKER_SYSTEM, RESEARCH_SYSTEM, ROUTER_SYSTEM, PLANNER_PROMPT

load_dotenv()


# pydantic models
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section"
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3-5 concrete, non-overlapping subpoints to cover in this section"
    )
    target_words: int = Field(
        ...,
        description="Target word count for this section (120-450)"
    )
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str = Field(..., description="who this blog id for")
    tone: str = Field(..., description="Writing tone (e.g., practical, crisp)")
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# State
class State(TypedDict):
    topic: str

    # will get from routing/research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # will get from worker
    sections: Annotated[List[tuple[int,str]], operator.add]
    final: str


# router
def router_node(state: State) -> dict:
    topic = state["topic"]
    decider = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content":
                ROUTER_SYSTEM
            },
            {
                "role": "user",
                "content":
                f"Topic: {topic}"
            }
        ],
        format=RouterDecision.model_json_schema()
    )

    decision= RouterDecision.model_validate_json(decider.message.content)

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries
    }


# route decision
def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "planner"


# research
def research_node(state: State) -> dict:
    queries = (state.get("queries", []) or [])
    MAX_RESULTS = 6

    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(_tavily_search(q,max_results= MAX_RESULTS))

    if not raw_results:
        return {"evidence": []}

    extractor = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": RESEARCH_SYSTEM
            },
            {
                "role": "user",
                "content": f"Raw Results:\n{raw_results}"
            },
        ],
        format=EvidencePack.model_json_schema()
    )

    pack = EvidencePack.model_validate_json(extractor.message.content)

    deduplicate = {}
    for e in pack.evidence:
        if e.url:
            deduplicate[e.url] = e

    return {"evidence": list(deduplicate.values())}


# tavily search
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    print("need to search tavily.")
    print("searching...")
    tool = TavilySearchResults(max_results = max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    print("found results")

    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date" or r.get("published_at")),
                "source": r.get("source"),
            }
        )

    return normalized


# planner
def planner(state: State) -> dict:
    print("called planner...")

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    msg = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": PLANNER_PROMPT
                    
            },
            {
                "role": "user",
                "content": f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
            }
        ],
        format=Plan.model_json_schema()
    )
    print("planner working...")
    plan = Plan.model_validate_json(msg.message.content)
    print("planner completed...")
    return {"plan": plan}


# intermediate function
def fanout(state: State):
    print("fanout called...")
    return [Send(
        "worker",
        {
            "task": task,
            "topic": state["topic"],
            "plan": state["plan"],
            "mode": state["mode"],
            "evidence": [e.model_dump() for e in state.get("evidence", [])]
        }
    )
    for task in state["plan"].tasks
    ]


# worker - generate content for one blog section
def worker(payload: dict) -> dict:
    print("worker called...")
    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    mode = payload.get("mode", "closed_book")


    blog_title = plan.blog_title
    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content":
                    WORKER_SYSTEM

            },
            {
                "role":"user",
                "content": 
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
            }
        ]
    )

    section_md = response.message.content.strip()
    print("sections ready...")
    return {"sections": [section_md]}


# join results of all workers to a complete blog, and saves it as a markdown file
def reducer(state: State) -> dict:
    print("reducer called...")
    title = state["plan"].blog_title
    body = "\n\n".join(state["sections"]).strip()

    final_md = f"# {title}\n\n{body}"
    print("final_md ready")

    # save to a file
    filename = title.lower().replace(" ", "_") + ".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")
    print("file ready")

    return {"final": final_md}



graph = StateGraph(State)


# graph - nodes
graph.add_node("planner", planner)
graph.add_node("worker", worker)
graph.add_node("reducer", reducer)
graph.add_node("research", research_node)
graph.add_node("router", router_node)


# graph - edges
graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    route_next,
    {
        "research": "research",
        "planner": "planner"
    }
)
graph.add_edge("research", "planner")
graph.add_conditional_edges(
    "planner",
    fanout,
    ["worker"]
)
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)


blog = graph.compile()


# invoke
out = blog.invoke({"topic": "Write a blog on AI in 2026", "section": []})