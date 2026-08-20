import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1. Load and split the document (same as before)
loader = PyPDFLoader("trip_plan.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

print(f"Split into {len(chunks)} chunks")

# 2. Set up the embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GEMINI_API_KEY"]
)

# 3. Build the vector store — this embeds every chunk and saves to disk
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector store built and saved to ./chroma_db")

# 4. Try a real search — no LLM involved yet, just retrieval
query = "How much does the Oregon trip cost?"
results = vectorstore.similarity_search(query, k=2)

print(f"\n--- Top 2 matches for: '{query}' ---")
for i, doc in enumerate(results):
    print(f"\nMatch {i+1}:")
    print(doc.page_content)