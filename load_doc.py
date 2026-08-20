from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the PDF — each page becomes a separate "Document" object
loader = PyPDFLoader("trip_plan.pdf")
pages = loader.load()

print(f"Loaded {len(pages)} pages")

# 2. Split into smaller chunks for better retrieval precision
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # roughly how many characters per chunk
    chunk_overlap=50     # overlap so context isn't lost at chunk boundaries
)
chunks = splitter.split_documents(pages)

print(f"Split into {len(chunks)} chunks")

# 3. Peek at the first chunk to see what one actually looks like
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk.page_content)