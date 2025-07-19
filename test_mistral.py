import fitz  # PyMuPDF
import docx
import requests


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def seach_and_summarize(query, db_path="faiss_index"):
    # Load FAISS
    vectorstore = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

    # Define the LLM
    llm = OllamaLLM(model="mistral")

    # Create LangChain RetrievalQA pipeline
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Get AI-generated answer
    response = qa_chain.invoke(query)

    print("\n🤖 AI-powered answer:")
    print(response["result"])



def generate_ai_response(context, query):
    prompt = f"""
You are an AI assistant with access to the following information:

{context}

Based on this, answer the following question:
{query}
"""

    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "No response generated.")
    except Exception as e:
        return f"❌ Error generating response: {e}"


def search_and_generated_response(query, db_path="faiss_index"):
    try:
        vectorstore = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
        results = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in results])
        ai_response = generate_ai_response(context, query)

        print("\n🤖 AI-powered answer:")
        print(ai_response)

    except Exception as e:
        print(f"❌ Error in FAISS search or Ollama call: {e}")


def process_document(file_path):
    text = extract_text(file_path)
    if not text:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    return texts


def store_embeddings(texts, db_path="faiss_index"):
    print("🧠 Storing embeddings in FAISS...")
    vectorstore = FAISS.from_texts(texts, embedding_model)
    vectorstore.save_local(db_path)
    print("✅ Embeddings stored and saved to disk.")


def search_documents(query, db_path="faiss_index"):
    print("🔍 Searching documents...")
    try:
        vectorstore = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
        results = vectorstore.similarity_search(query, k=5)

        if not results:
            print("⚠️ No matching documents found.")
            return

        for idx, result in enumerate(results):
            print(f"\n🔹 Result {idx + 1}:")
            print(result.page_content.strip()[:500])

        return results

    except Exception as e:
        print("❌ Error loading FAISS index or performing search:", e)


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text


def extract_text_from_docx(doc_path):
    text = ""
    try:
        doc = docx.Document(doc_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {doc_path}: {e}")
    return text


def extract_text_from_txt(txt_path):
    text = ""
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading TXT {txt_path}: {e}")
    return text


def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        return None


if __name__ == "__main__":
    sample_file = "sample.pdf" 
    texts = process_document(sample_file)

    if texts:
        store_embeddings(texts)
    else:
        print("⚠️ No text found in the document.")

    user_query = input("Enter your search query: ")
    if user_query.strip():
        #search_and_generated_response(user_query)
        seach_and_summarize(user_query)
    else:
        print("⚠️ No query entered.")
