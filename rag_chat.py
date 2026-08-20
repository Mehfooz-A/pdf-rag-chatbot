import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 1. Load the existing vector store (already built — no re-embedding needed)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GEMINI_API_KEY"]
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 2. Turn the vectorstore into a retriever — searches automatically when called
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. The model, same as before
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=os.environ["GEMINI_API_KEY"]
)

# 4. A prompt template that expects BOTH retrieved context and the question
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question:
{question}
""")

# 5. A helper to turn retrieved chunks into one text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. The full RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 7. Ask it something real
print("RAG chatbot ready. Ask about the trip plan document.\n")

while True:
    question = input("Ask something (or type 'quit' to exit): ")
    if question.lower() == "quit":
        break
    answer = rag_chain.invoke(question)
    print(f"\n{answer}\n")