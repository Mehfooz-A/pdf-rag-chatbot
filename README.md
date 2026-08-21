Chat with your PDF — RAG Chatbot



A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask questions about it in plain English. Answers are grounded in the document's actual content, with source page citations — not general knowledge guessing.





\[Try it live →](https://pdf-rag-chatbot-dygeu2uxt3eppj7n9orr7o.streamlit.app/)





What it does



\- Upload any PDF document

\- Ask questions in natural language

\- Get answers generated from the document's actual content, with page-number citations

\- Handles corrupted files, scanned-image PDFs, and API rate limits gracefully





Tech stack



\- LangChain\*\* — orchestration (prompt templates, retrieval chains)

\- Google Gemini API\*\* — LLM (`gemini-3.6-flash`) + embeddings (`gemini-embedding-001`)

\- ChromaDB\*\* — vector store for semantic search

\- Streamlit\*\* — web interface, deployed on Streamlit Cloud





How it works



1\. PDF is loaded and split into overlapping text chunks

2\. Each chunk is converted into an embedding (a numerical representation of its meaning)

3\. Embeddings are stored in ChromaDB for fast similarity search

4\. On a question, the most relevant chunks are retrieved and passed to the LLM as context

5\. The model answers using only that retrieved context — reducing hallucination and enabling source citations





Run it locally



```bash

git clone https://github.com/Mehfooz-A/pdf-rag-chatbot.git

cd pdf-rag-chatbot

pip install -r requirements.txt

```



Create a `.env` file with your own Gemini API key:

GEMINI_API_KEY=your_key_here



Then run:

```bash

streamlit run app.py

```



Deep dive



For a full explanation of the architecture, design decisions, and how RAG works under the hood, see \[ARCHITECTURE.md](./ARCHITECTURE.md).





Notes



\- Uses Google Gemini's free tier — large documents may take longer to process due to API rate limits (handled gracefully with batching).

\- This is Module 1 of a larger project — an HR Intelligence Suite combining document Q\&A, attrition prediction, and BI reporting for SMEs.

