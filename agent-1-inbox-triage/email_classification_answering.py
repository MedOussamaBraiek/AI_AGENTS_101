from dotenv import load_dotenv
from typing import Literal, TypedDict, NotRequired
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

load_dotenv("./.env")

class ClassificationResult(BaseModel):
    classification: Literal["urgent", "needs_reply", "spam", "fyi"]
    justification: str = Field(description="Why you classified it that way")
    score: float = Field(description="Your confidence for the classification from 0 to 1")

class GraphState(TypedDict):
    text: str
    classification: NotRequired[Literal["urgent","needs_reply","spam","fyi"]] 
    justification: NotRequired[str] 
    text_summary: NotRequired[str] 
    draft: NotRequired[str] 
    human_decision: NotRequired[Literal["approve","edit","reject"]]
    edited_version: NotRequired[str] 

# Get the LLM
llm = ChatGroq(model="openai/gpt-oss-120b")

# Declaring nodes
def email_classifier(state: GraphState) -> GraphState:
    """This node classify the text in the state as urgent or needs_reply or spam or fyi"""
    classifier = llm.with_structured_output(ClassificationResult)
    result = classifier.invoke("Classify this email as urgent or needs_reply or spam or fyi: " + state["text"])
    return {
        "text": state["text"],
        "classification": result.classification,
        "justification": result.justification,
        "score": result.score
    }

def text_summary(state: GraphState) -> GraphState:
    """This node summarize the text in the state"""
    result = llm.invoke(f"Read this text input: {state["text"]} and summarize it").content
    return {   
        "text_summary": result
        }

def draft_email(state: GraphState) -> GraphState:
    """This node draft an email from a giving text"""
    print("This is the draft email node")

    if state.get("edited_version"):
        prompt = f"The user gave this feedback on the previous draft: '{state['edited_version']}'. Re-draft the email to incorporate their feedback. Original email topic: {state['text']}"
    else:
        prompt = f"Draft a professional email response to: {state['text']}"
    
    result = llm.invoke(prompt).content

    return {
        "draft": result,
        "classification": state.get("classification"),  
        "justification": state.get("justification"),   
        "score": state.get("score"),                   
    }

def interrupt_node(state: GraphState) -> GraphState:
    """Get human feedback"""
    print(f"\n--- DRAFT ---\n{state['draft']}\n")
    feedback = input("Approve (a), Edit (e), or Reject (r)? ").lower()

    decision_map = {"a": "approve", "e": "edit", "r": "reject"}
    human_decision = decision_map.get(feedback, "approve")
    
    edited = ""
    if human_decision == "edit":
        edited = input("Paste edited version: ")
    
    return {
        "human_decision": human_decision,
        "edited_version": edited
    }

def send_email(state: GraphState) -> GraphState:
    """This node takes the edited text from the interrupt or if that not provided takes the draft and send an email"""
    if state.get("edited_version"):
        print("Send email with edited version")
    else:
        print("Send email with draft version")
    return state

def router(state: GraphState) -> GraphState:
    decission = state["classification"]
    if decission == "spam":
        return "ignore"
    elif decission == "fyi":
        return "summary"
    else:
        return "respond"

def should_continue(state: GraphState) -> str:
    feedback = state["human_decision"]
    if feedback == "reject":
        return "end"
    elif feedback == "approve":
        return "accept"
    else:
        return "edit"

# Building the Graph
graph = StateGraph(GraphState)

graph.add_node("classification_node", email_classifier)
graph.add_node("summarization_node", text_summary)
graph.add_node("drafting_node", draft_email)
graph.add_node("interruption_node", interrupt_node)
graph.add_node("sending_node", send_email)

graph.add_edge(START, "classification_node")
graph.add_edge("drafting_node", "interruption_node")

graph.add_conditional_edges(
    "classification_node",
    router,
    {
        "summary" : "summarization_node",
        "respond" : "drafting_node",
        "ignore" : END,
    }
)

graph.add_conditional_edges(
    "interruption_node",
    should_continue,
    {
        "accept" : "sending_node",
        "edit" : "drafting_node",
        "end" : END
    }
)
graph.add_edge("summarization_node", END)
graph.add_edge("sending_node", END)

tests = [
"URGENT: Server is down! Production is completely offline. We need immediate action!",
"Hi, just letting you know the team lunch is at 12pm tomorrow in the main conference room.",
"Congratulations! You've won $1,000,000! Claim your prize now by clicking here.",
"Can you review the design doc I sent yesterday? Need your feedback by EOD."
]

if __name__ == "__main__":
    compiled_graph = graph.compile()
    print("Graph compiled successfully!")

    graph_image = compiled_graph.get_graph().draw_mermaid_png()
    with open("graph_visualization.png", "wb") as f:
        f.write(graph_image)
    print("Graph saved to graph_visualization.png")

    for i, test in enumerate(tests):
        print(f"\n=== Test {i+1} ===")
        result = compiled_graph.invoke({"text": test})
        print(f"Classification: {result.get('classification')}")
        print(f"Score: {result.get('score')}")
        if result.get("text_summary"):
            print(f"Summary: {result.get('text_summary')}")
        if result.get("draft"):
            print(f"Draft: {result.get('draft')[:100]}...")
    print("DONE!!!")
