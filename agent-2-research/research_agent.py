import os
import json
from dotenv import load_dotenv
from typing import Literal, TypedDict, NotRequired, List
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv("./.env")

# Get the LLM
llm = ChatGroq(model="openai/gpt-oss-120b")

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


class GraphState(TypedDict):
    text: str
    search_query: NotRequired[str]
    response: NotRequired[str] 
    needs_more_info: NotRequired[bool]
    queries: NotRequired[List[str]]
    resources: NotRequired[List[str]]
    iteration: NotRequired[int]
    max_iteration: NotRequired[int]


def think_node(state: GraphState) -> GraphState:
    """This nodes understand the query from the user and decides what to search for"""
    prompt = f"Question: {state['text']}\n\nRespond with ONLY a short search query (max 10 words), no explanation."
    result = llm.invoke(prompt).content.strip()
    return {
        "search_query": result,
        "queries": (state.get("queries") or []) + [result]  
    }

def search_node(state: GraphState) -> GraphState:
    """This node uses the search tool to get results and resources"""
    query = state["search_query"]
    results = tavily.search(query)

    # results is a dict with "results" key containing list of search results
    # Each result has: title, url, content
    
    return {
        "response": results, 
        "resources": [r["url"] for r in results["results"]]  
    }

def evaluation_node(state: GraphState) -> GraphState:
    """This node evaluate the result from the search tool against the user query and decides if it needs more info or not"""
    prompt = f"""
    Question: {state["text"]}
    Search results: {state["response"]}
    Previous searches: {state["queries"]}
    
    Do you have enough info to answer? 
    Answer with JSON: {{"answer_ready": true/false, "final_answer": "...", "next_search": "..."}}
    """
    result = llm.invoke(prompt).content
    data = json.loads(result)
    return {
        "needs_more_info": not data.get("answer_ready", False),
        "response": data.get("final_answer", state["response"]),
        "queries": state["queries"] + (data.get("next_search", "") if isinstance(data.get("next_search"), list) else [data.get("next_search", "")]),
        "iteration": (state.get("iteration") or 0) + 1
    }

def format_node(state: GraphState) -> GraphState:
    """This nodes takes the result of the search and the resources used and return a readable response to the user"""
    prompt = f"""
        Question: {state["text"]}
        Search results: {state["response"]}
        Resource: {state["resources"]}
        
        Look at the question and answer it with the result from the search and include resources
        """
    result = llm.invoke(prompt).content
    return {
        "response": result
    }



def should_continue(state: GraphState) -> str:
    if state['needs_more_info']:
        return "think"
    else:
        return "format"


graph = StateGraph(GraphState)

graph.add_node("think_node", think_node)
graph.add_node("search_node", search_node)
graph.add_node("evaluation_node", evaluation_node)
graph.add_node("format_node", format_node)

graph.add_edge(START, "think_node")
graph.add_edge("think_node", "search_node")
graph.add_edge("search_node", "evaluation_node")

graph.add_conditional_edges(
    "evaluation_node",
    should_continue,
    {
        "think" : "think_node",
        "format" : "format_node"
    }
)

graph.add_edge("format_node", END)


if __name__ == "__main__":
    compiled_graph = graph.compile()
    print("Graph compiled successfully!")

    tests = [
        "What is artificial intelligence?",
        "Latest Python programming trends 2026",
        "How does blockchain work?"
    ]
    
    for q in tests:
        print(f"\n=== Question: {q} ===")
        result = compiled_graph.invoke({"text": q})
        print(f"\nQuery: {result['search_query']}")
        print(f"\nAnswer: {result['response'][:200]}...")
        print(f"\nSources: {result.get('resources', [])[:3]}")