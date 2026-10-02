import json
from dotenv import load_dotenv
from typing import TypedDict, NotRequired, List
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv("./.env")

checkpointer = MemorySaver()

# Load the LLM
llm = ChatGroq(model="openai/gpt-oss-120b")


class GraphState(TypedDict):
    user_id: NotRequired[str]     
    weak_topics: NotRequired[List[str]]      
    score_history: NotRequired[List[int]]                             
    topic: NotRequired[str]                                     
    question: NotRequired[str]                                
    user_answer: NotRequired[str]                               
    score: NotRequired[int]           
    explanation: NotRequired[str]   
    should_continue: NotRequired[bool]   


store = {}

def get_store(key):
    return store.get(key)

def put_store(key, value):
    store[key] = value        


def load_profile_node(state: GraphState) -> GraphState:
    """Load user profile from store"""
    user_id = state.get("user_id", "default_user")

    weak_topics = store.get(("user", user_id, "weak_topics")) or [] # user namespace for user_id
    score_history = store.get(("user", user_id, "score_history")) or []

    return {
        "user_id": user_id,
        "weak_topics": weak_topics,      
        "score_history": score_history  
    }

def select_topic_node(state: GraphState) -> GraphState:
    """This node will select the topic"""
    prompt="""You are an expert study coach. Quiz users, evaluate answers, identify weak spots, adjust difficulty"""

    score_history = state.get("score_history", [])
    weak_topics = state.get("weak_topics", [])
    last_score = score_history[-1] if score_history else 100

    # If has weak topics AND last score was low, focus on weak areas
    if weak_topics and last_score <= 70:
        prompt = f"Pick ONE topic from this weak areas list to help improve: {weak_topics}. Return ONLY the topic name."
    else:
        # First time OR good score, pick any topic
        prompt = f"Pick one interesting programming topic to quiz on (recursion, decorators, OOP, APIs, etc). Return ONLY the topic name."
    
    result = llm.invoke(prompt).content.strip()
    
    return {"topic": result}

def ask_question_node(state: GraphState) -> GraphState:
    """Generate a quiz question based on topic and difficulty"""
    score_history = state.get("score_history", [])
    last_score = score_history[-1] if score_history else 100  
    difficulty = "harder" if last_score > 70 else "easier"

    prompt = f"""
    Topic: {state['topic']}
    Difficulty: {difficulty}
    
    Generate a quiz question on {state['topic']}.
    If difficulty is 'harder', make it challenging.
    If difficulty is 'easier', make it beginner-friendly.
    Return ONLY the question, nothing else.
    """
    result = llm.invoke(prompt).content.strip()
    return {
         "question": result
    }

def answer_node(state: GraphState) -> GraphState:
    """Wait for user input"""
    print(f"\n📚 Question: {state['question']}\n")
    user_answer = input("Your answer: ").strip()
    
    return {"user_answer": user_answer}

def evaluate_answer_node(state: GraphState) -> GraphState:
    """Evaluate user answer"""

    prompt = f"""
        Question: {state['question']}
        Answer: {state['user_answer']}
        
        Evaluate the answer on that question and give it a score.
        Return ONLY the score, nothing else.
        """
    result = llm.invoke(prompt).content.strip()
    return {"score": result}

def update_profile_node(state: GraphState) -> GraphState:
    """This node will update the user profile"""
    store.get(("user", state['user_id'], "score_history"))
    return state

def should_continue(state: GraphState) -> str:
    """This node will decide continue or end"""
    if state.get('should_continue', False):
        return 'continue'
    else:
        return 'end'

graph = StateGraph(GraphState)

graph.add_node("load_profile", load_profile_node)
graph.add_node("select_topic", select_topic_node)
graph.add_node("ask_question", ask_question_node)
graph.add_node("answer_node", answer_node)
graph.add_node("evaluate_answer", evaluate_answer_node)
graph.add_node("update_profile", update_profile_node)

graph.add_edge(START, "load_profile")
graph.add_edge("load_profile", "select_topic")
graph.add_edge("select_topic", "ask_question")
graph.add_edge("ask_question", "answer_node")
graph.add_edge("answer_node", "evaluate_answer")
graph.add_edge("evaluate_answer", "update_profile")

graph.add_conditional_edges(
    "update_profile",
    should_continue,
    {
        "continue": "ask_question",
        "end": END
    }
)

if __name__ == "__main__":
    compiled_graph = graph.compile(checkpointer=checkpointer)
    print("Graph compiled successfully!")

    graph_image = compiled_graph.get_graph().draw_mermaid_png()
    with open("graph_visualization.png", "wb") as f:
                f.write(graph_image)
    print("Graph saved to graph_visualization.png")

    result = compiled_graph.invoke(
    {"user_id": "oussama", "question": "Test"},
    config={"configurable": {"thread_id": "user_123"}}
    )
    print(result)