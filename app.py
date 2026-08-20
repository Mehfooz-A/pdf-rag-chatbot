import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="Chat with your PDF", page_icon=":page_facing_up:")
st.title("Chat with your PDF")
st.caption("Upload a PDF and ask questions about it - powered by RAG (LangChain + Gemini + ChromaDB)")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.button("Process document"):
        with st.spinner("Reading and indexing your PDF..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                pages = loader.load()

                if not pages:
                    st.error("Could not read any text from this PDF - it may be a scanned image without extractable text.")
                    st.stop()

                if len(pages) > 30:
                    st.warning(f"This PDF has {len(pages)} pages. Large documents may take longer due to free-tier API rate limits.")

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(pages)

                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=os.environ["GEMINI_API_KEY"]
                )

                BATCH_SIZE = 90
                vectorstore = None

                for i in range(0, len(chunks), BATCH_SIZE):
                    batch = chunks[i:i + BATCH_SIZE]
                    if vectorstore is None:
                        vectorstore = Chroma.from_documents(documents=batch, embedding=embeddings)
                    else:
                        vectorstore.add_documents(batch)

                    if i + BATCH_SIZE < len(chunks):
                        st.info(f"Indexed {i + len(batch)}/{len(chunks)} chunks - pausing briefly to respect API rate limits...")
                        time.sleep(65)

                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                st.session_state.retriever = retriever

                model = ChatGoogleGenerativeAI(
                    model="gemini-3.6-flash",
                    api_key=os.environ["GEMINI_API_KEY"]
                )
                prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question:
{question}
""")

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                st.session_state.rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | model
                    | StrOutputParser()
                )
                st.session_state.messages = []

                st.success(f"Ready! {len(chunks)} chunks indexed from {uploaded_file.name}")

            except Exception as e:
                st.error(f"Something went wrong processing this PDF: {e}")

    if st.session_state.rag_chain is not None:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

if st.session_state.rag_chain is None:
    st.info("Upload a PDF in the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.caption(f"Page {src['page']}: {src['snippet']}")

    question = st.chat_input("Ask something about your document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    docs = st.session_state.retriever.invoke(question)
                    answer = st.session_state.rag_chain.invoke(question)
                    st.write(answer)

                    sources = [
                        {
                            "page": doc.metadata.get("page", "?") + 1,
                            "snippet": doc.page_content[:150] + "..."
                        }
                        for doc in docs
                    ]
                    with st.expander("Sources"):
                        for src in sources:
                            st.caption(f"Page {src['page']}: {src['snippet']}")

                    st.session_state.messages.append({
                        "role": "assistant", "content": answer, "sources": sources
                    })
                except Exception as e:
                    error_msg = f"Something went wrong answering that: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
