from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from parser_agent import ParsingResult

load_dotenv("./.env")

llm = ChatGroq(model="openai/gpt-oss-120b")

class ScoreResult(BaseModel):
    match_score: int = Field(description="Score 0-100 based on CV match")
    gaps: List[str] = Field(description="List of gaps missing from CV based on the Job if exist")
    strengths: List[str] = Field(description="List of strengths from CV based on the Job if exist")

class ScorerState(TypedDict):
    parsed_cv: str
    job_posting: str
    score : ScoreResult  


def scorer_node(state: ScorerState) -> ScorerState:
    """Score the CV based on a giving Job description"""

    parsed_cv_dict = state['parsed_cv'].model_dump()
    
    prompt=f"""
    CV Data: {parsed_cv_dict}
    Job: {state['job_posting']}

    Score the CV based on the giving Job description and return ONLY valid JSON (no markdown).
"""
    scorer = llm.with_structured_output(ScoreResult)
    result = scorer.invoke(prompt)

    return {"score": result}

scorer_graph = StateGraph(ScorerState)

scorer_graph.add_node("score", scorer_node)

scorer_graph.add_edge(START, "score")
scorer_graph.add_edge("score", END)

compiled_scorer = scorer_graph.compile()

if __name__ == "__main__":
    # Mock parsed CV
    mock_cv = ParsingResult(
        skills=["Python", "FastAPI"],
        experience=["2 years backend"],
        education=["BS CS"],
        languages=["English"],
        certificates=[],
        name="Oussama Braiek",
        email="oussama@gmail.com",
        phone="89567234"
    )
    
    job = "Senior Python dev. Need: Django, 5yr exp, AWS, Docker"
    
    result = compiled_scorer.invoke({
        "parsed_cv": mock_cv,
        "job_posting": job
    })
    print(result)