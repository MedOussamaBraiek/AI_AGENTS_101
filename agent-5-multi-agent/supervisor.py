from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from parser_agent import compiled_parser, ParsingResult
from scorer_agent import compiled_scorer
from letter_agent import compiled_letter

class SupervisorState(TypedDict):
    job_posting: str
    cv_text: str
    parsed_cv: ParsingResult
    match_score: int
    gaps: List[str]
    strengths: List[str]
    cover_letter: str
    draft_email: str
    human_approved: bool

load_dotenv("./.env")

llm = ChatGroq(model="openai/gpt-oss-120b")

def run_parser(state):
    # Call parser_agent
    result = compiled_parser.invoke({"cv_text": state["cv_text"]})
    return {"parsed_cv": result["parsed_cv"]}

def run_scorer(state):
    # Call scorer_agent
    result = compiled_scorer.invoke({"parsed_cv": state["parsed_cv"], "job_posting": state["job_posting"]})
    score_result = result["score"]
    return {
        "match_score": score_result.match_score,
        "gaps": score_result.gaps,
        "strengths": score_result.strengths
    }

def run_letter(state):
    # Call letter_agent
    result = compiled_letter.invoke({"job_posting": state["job_posting"], "gaps": state["gaps"], "strengths": state["strengths"]})
    letter_result = result["result"]
    return {
        "cover_letter": letter_result.cover_letter,
        "draft_email": letter_result.draft_email
    }

def approve_node(state):
    print(f"Match score: {state['match_score']}")
    print(f"Cover letter:\n{state['cover_letter']}")
    approve = input("Approve? (y/n): ").lower() == 'y'
    return {"human_approved": approve}


def should_generate_letter(state):
    """Route based on match score"""
    if state["match_score"] >= 50: 
        return "generate_letter"
    else:
        return "low_match"

def low_match_node(state):
    """Handle low match - guide user"""
    print(f"\n⚠️  Match Score: {state['match_score']}/100 (Low)")
    print(f"\nGaps to address:")
    for gap in state["gaps"]:
        print(f"  • {gap}")
    print(f"\nYour Strengths:")
    for strength in state["strengths"]:
        print(f"  • {strength}")
    print("\n💡 Recommendation: Consider building skills in the gap areas before applying.")
    return { "cover_letter": "Not generated",
        "draft_email": "",
        "human_approved": False}


supervisor_graph = StateGraph(SupervisorState)

supervisor_graph.add_node("parser", run_parser)
supervisor_graph.add_node("scorer", run_scorer)
supervisor_graph.add_node("writer", run_letter)
supervisor_graph.add_node("approve", approve_node)
supervisor_graph.add_node("low_match", low_match_node)

supervisor_graph.add_edge(START, "parser")
supervisor_graph.add_edge("parser", "scorer")
supervisor_graph.add_conditional_edges(
    "scorer",
    should_generate_letter,
    {
        "generate_letter": "writer",
        "low_match": "low_match"
    }
)
supervisor_graph.add_edge("writer", "approve")
supervisor_graph.add_edge("low_match", END)
supervisor_graph.add_edge("approve", END)

compiled_supervisor = supervisor_graph.compile()


if __name__ == "__main__":
    print("Graph compiled successfully!")

    graph_image = compiled_supervisor.get_graph().draw_mermaid_png()
    with open("graph_visualization.png", "wb") as f:
                    f.write(graph_image)
    print("Graph saved to graph_visualization.png")
    
    # Test 1: Good match
    test_cv_1 = "Oussama Braiek, Python dev with 5yr exp. Skills: Python, Django, AWS, Docker. Education: BS CS. Email: oussama@gmail.com"
    job_1 = "Senior Python dev. Need: Django, AWS, Docker, 5yr exp"
    
    # Test 2: Low match
    test_cv_2 = "Junior JavaScript developer. Skills: JS, React. Education: Boot camp"
    job_2 = "Senior Python dev. Need: Django, AWS, Docker, 5yr Python"
    
    # Test 3: Medium match
    test_cv_3 = "Python dev, 3 years. Skills: Python, FastAPI. Education: BS CS"
    job_3 = "Python Backend. Need: Django, 5yr exp, PostgreSQL"
    
    for i, (cv, job) in enumerate([(test_cv_1, job_1), (test_cv_2, job_2), (test_cv_3, job_3)], 1):
        print(f"\n{'='*60}\nTest {i}")
        result = compiled_supervisor.invoke({"cv_text": cv, "job_posting": job})
        print(f"Match Score: {result['match_score']}/100")
        print(f"Approved: {result['human_approved']}")