import json
from dotenv import load_dotenv
from typing import TypedDict, NotRequired, List
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pdfplumber


load_dotenv("./.env")

# Load the LLM
llm = ChatGroq(model="openai/gpt-oss-120b")

# Load embeddings (free, local)
embeddings = HuggingFaceEmbeddings()

# Load PDF
# Documents is a list of strings (one per page)
documents = []
with pdfplumber.open("./resume_en.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:  # only add non-empty pages
            documents.append(text)

class GraphState(TypedDict):
    question: str
    query_rewritten: str
    pdf_chunks: NotRequired[List[str]]
    retrieved_chunks: NotRequired[List[str]]
    retrieval_grade: NotRequired[int]
    num_chunks_to_retrieve: NotRequired[int]
    final_answer: NotRequired[str] 
    iteration: NotRequired[int]
    max_iteration: NotRequired[int]

def chunk_pdf(documents, chunk_size=500, overlap=100):
    # 1. Combine all page text into one string
    full_text = "\n".join(documents)
    
    # 2. Split into chunks of chunk_size with overlap
    chunks = []
    for i in range(0, len(full_text), chunk_size - overlap): # step = 400
        chunk = full_text[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks


chunks = chunk_pdf(documents)
print(f"Number of pages: {len(documents)} / Number of chunks: {len(chunks)}")

def retriever_node(state: GraphState) -> GraphState:
    """Retrieve most simular chunks based on question"""
   
    query = state.get("query_rewritten", state["question"])
    
    # Create vector store from chunks
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Retrieve top chunks matching the question
    retrieved = vectorstore.similarity_search(query, k=3)
    # Extract text from retrieved documents
    retrieved_texts = [doc.page_content for doc in retrieved]

    return {
        "retrieved_chunks": retrieved_texts
    }

def evaluation_node(state: GraphState) -> GraphState:
    """Evaluate the chunks retrieved based on the question"""
    prompt=f"""
    Question : {state["question"]}
    Retrieved chunks : {state['retrieved_chunks']}

    Grade these chunks (0-1): Are they relevant to answer the question?
    If grade < 0.7, suggest a better search query.
    
    JSON: {{"grade": 0.8, "rewritten_query": "better query or null"}}
    """
    result = llm.invoke(prompt).content
    data = json.loads(result)
    return {
        "retrieval_grade": data.get("grade", 0),
        "query_rewritten": data.get("rewritten_query", state["question"])
    }

def format_node(state: GraphState) -> GraphState:
    """Generate final answer from retrieved chunks"""
    prompt = f"""
    Question: {state["question"]}
    Context from document: {state["retrieved_chunks"]}
    
    Answer the question using the context. Be concise.
    """
    answer = llm.invoke(prompt).content
    return {
        "final_answer": answer
    }


def should_continue(state: GraphState) -> str:
    if state["retrieval_grade"] >= 0.7:
        return 'format'
    else:
        return 'rewrite'


# Build the Graph
graph = StateGraph(GraphState)

graph.add_node('retriever', retriever_node)
graph.add_node('evaluator', evaluation_node)
graph.add_node('formatter', format_node)

graph.add_edge(START, 'retriever')
graph.add_edge('retriever', 'evaluator')

graph.add_conditional_edges(
    'evaluator',
    should_continue,
    {
        'rewrite': 'retriever',
        'format': 'formatter'
    }
)

graph.add_edge('formatter', END)

# Questions for testing    
questions = [
    "What are your skills?",
    "What's your experience with AI?",
    "Tell me about your full-stack development experience"
]

if __name__ == "__main__":
    compiled_graph = graph.compile()
    print("Graph compiled successfully!")

    graph_image = compiled_graph.get_graph().draw_mermaid_png()
    with open("graph_visualization.png", "wb") as f:
                f.write(graph_image)
    print("Graph saved to graph_visualization.png")

    for q in questions:
        print(f"\n=== {q} ===")
        result = compiled_graph.invoke({"question": q})
        print(result["final_answer"])