from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

#initialize FastAPI app
app = FastAPI()


#load mistral AI via ollama
llm = OllamaLLM(model="mistral")

#load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#load the FAISS vector store
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

#define the request model
class QueryRequest(BaseModel):
    query: str

@app.post("/query")

def search_and_generate_response(request: QueryRequest):
    #retrieve documents and generate AI-powered response

    response = qa_chain.invoke(request.query)
    return{"query": request.query, "response": response}

#root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the AI-powered search API!"} 