import json
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

load_dotenv("./.env")

llm = ChatGroq(model="openai/gpt-oss-120b")

class ParsingResult(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = Field(description="List of skills from the CV if exist")
    languages: List[str] = Field(description="List of languages from CV if exist")
    experience: List[str] = Field(description="List of experiences from CV if exist")
    education: List[str] = Field(description="List of educations from CV if exist")
    certificates: List[str] = Field(description="List of certificates from CV if exist")

class ParserState(TypedDict):
    cv_text: str
    parsed_cv: ParsingResult  

def parser_node(state: ParserState) -> ParserState:
    """Parse CV text"""
    prompt = f"""
    Extract from this CV. Return ONLY valid JSON.

    CV: {state['cv_text']}

    JSON format (ALL fields must be lists):
    {{
    "name": "string or empty",
    "email": "string or empty",
    "phone": "string or empty",
    "skills": ["skill1", "skill2"],
    "languages": ["lang1", "lang2"],
    "experience": ["exp1", "exp2"],
    "education": ["edu1", "edu2"],
    "certificates": ["cert1", "cert2"]
    }}
    """
    result = llm.invoke(prompt).content.strip()
    
    try:
        data = json.loads(result)

        for field in ["skills", "languages", "experience", "education", "certificates"]:
            if field in data and isinstance(data[field], str):
                data[field] = [data[field]]  
            elif field not in data:
                data[field] = []  

        for field in ["name", "email", "phone"]:
            if field in data:
                if isinstance(data[field], list):
                    data[field] = data[field][0] if data[field] else ""
                elif not isinstance(data[field], str):
                    data[field] = ""
            else:
                data[field] = ""

        parsed = ParsingResult(**data)  
    except json.JSONDecodeError:
        parsed = ParsingResult(skills=[], languages=[], experience=[], education=[], certificates=[], name="", email="", phone="")
    
    return {"parsed_cv": parsed}


parser_graph = StateGraph(ParserState)

parser_graph.add_node("parse", parser_node)

parser_graph.add_edge(START, "parse")
parser_graph.add_edge("parse", END)

compiled_parser = parser_graph.compile()


if __name__ == "__main__":
    test_cv = "Oussama Braiek, Python dev with 5yr exp. Skills: Python, Django, AWS, Docker. Education: BS CS. Email: oussama@gmail.com"
    
    result = compiled_parser.invoke({"cv_text": test_cv})
    print(result["parsed_cv"])