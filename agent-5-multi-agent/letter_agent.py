from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

load_dotenv("./.env")

llm = ChatGroq(model="openai/gpt-oss-120b")

class LetterResult(BaseModel):
    cover_letter: str = Field(description="Professional cover letter")
    draft_email: str = Field(description="Draft email to send with letter")

class LetterState(TypedDict):
    job_posting: str
    gaps: List[str]
    strengths: List[str]
    result : LetterResult  

def letter_node(state: LetterState) -> LetterState:
    """Generate cover letter and drafting email"""
    prompt=f"""
    Job description: {state['job_posting']}
    gaps: {state["gaps"]}
    strengths: {state["strengths"]}

    Generate cover letter and professional email ready to be sent.
    return ONLY valid JSON (no markdown).
"""
    writer = llm.with_structured_output(LetterResult)
    result = writer.invoke(prompt)
    return {"result": result}


writer_graph = StateGraph(LetterState)

writer_graph.add_node("write", letter_node)

writer_graph.add_edge(START, "write")
writer_graph.add_edge("write", END)

compiled_letter = writer_graph.compile()


if __name__ == "__main__":
    test_result = compiled_letter.invoke({
        "job_posting": "Senior Python dev. Need: Django, AWS",
        "gaps": ["Django", "AWS", "5yr exp"],
        "strengths": ["Python"]
    })
    print(test_result["result"].cover_letter)
    print("\n---\n")
    print(test_result["result"].draft_email)