import os
import sys
import logging
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict

load_dotenv()
## load the GROQ API Key
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

## load HF API Key
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

# Langsmith Tracking and Langchain API Key
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

# Set up structured logging
logger = logging.getLogger("rag_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Loading Embedding Model
embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

loader=PyPDFDirectoryLoader("research_papers") ## Data Ingestion step
docs=loader.load() ## Document Loading
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Creating Vectorstore and retriever
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Loading LLama model from Groq API
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")

# Prompt 
prompt_template="""
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question

    <context>
    {context}
    <context>

    Question:{input}
    """

prompt = ChatPromptTemplate.from_template(prompt_template)

# Creating Document & Retrieval chain
document_chain=create_stuff_documents_chain(llm,prompt)
document_chain
retrieval_chain=create_retrieval_chain(retriever,document_chain)
retrieval_chain

# Initiating FastAPI
app = FastAPI()

# ------------------------------
# Request & Response Models
# ------------------------------

class QueryRequest(BaseModel):
    query: str

class ContextDoc(BaseModel):
    score: float
    metadata: Dict[str, Any]
    summary: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    embedding_model: str
    prompt_used: str
    context_documents: List[ContextDoc]

# ------------------------------
# Define endpoint
# ------------------------------

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    logger.info(f'{{"event": "api_request_received", "query": "{request.query}"}}')

    try:
         # Handle empty query early
        if not request.query.strip():
            logger.warning(f'{{"event": "bad_request", "reason": "empty_query"}}')
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        # Running retrieval chain
        rag_response = retrieval_chain.invoke({"input": request.query})
        answer_text = rag_response.get("answer", "")

        # Handling empty answers
        if not answer_text.strip():
            logger.warning(f'{{"event": "llm_failure", "reason": "empty_response"}}')
            raise HTTPException(status_code=502, detail="Language model returned an empty response.")

        logger.info(f'{{"event": "llm_response", "response": "{answer_text}"}}')

        # Retrieve similarity scores
        retriever_results = retriever.vectorstore.similarity_search_with_score(
            request.query,
            k=3
        )

        # Handling empty retrieval results
        if not retriever_results:
            logger.warning(f'{{"event": "retrieval_failure", "reason": "no_similar_documents"}}')
            raise HTTPException(status_code=404, detail="No relevant documents found for the query.")

        # Summary
        context_documents = []
        for doc, score in retriever_results:
            logger.info(f'{{"event": "retrieval_result", "score": {score}, "metadata": {doc.metadata}}}')
            context_documents.append(
                ContextDoc(
                    score=float(score),
                    metadata=doc.metadata,
                    summary=doc.page_content[:100] + "..."  # simple snippet as summary
                )
            )

        return QueryResponse(
            query=request.query,
            answer=answer_text,
            embedding_model="BAAI/bge-base-en-v1.5",
            prompt_used=prompt_template.strip(),
            context_documents=context_documents
        )
    
    except HTTPException as he:
        raise he

    except Exception as e:
        logger.error(f'{{"event": "api_error", "error": "{str(e)}"}}')
        raise HTTPException(status_code=500, detail=str(e))