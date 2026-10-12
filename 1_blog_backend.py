from __future__ import annotations
import operator
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import ollama
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from model import MODEL


# pydantic models
class Task(BaseModel):
    id: int
    title: str
    brief: str = Field(..., description="What to cover")


class Plan(BaseModel):
    blog_title: str
    tasks: List[Task]


# State
class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]  # joins the results provided by all tasks
    final: str


# planner
def planner(state: State) -> dict:
    print("called planner...")
    msg = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": f"Create a blog plan with 5-7 sections on the following topic."
            },
            {
                "role": "user",
                "content": f"Topic: {state["topic"]}"
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
            "plan": state["plan"]
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

    blog_title = plan.blog_title

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content":
                "Write one clean markdown section"

            },
            {
                "role":"user",
                "content": f"Blog: {blog_title}\n"
                f"Topic: {topic}\n"
                f"Section: {task.title}\n"
                f"Brief: {task.brief}\n"
                "Return only the section content in Markdown"
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


# graph - edges
graph.add_edge(START, "planner")
graph.add_conditional_edges("planner", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)


blog = graph.compile()


# invoke
out = blog.invoke({"topic": "Write a blog on Self Attention", "section": []})