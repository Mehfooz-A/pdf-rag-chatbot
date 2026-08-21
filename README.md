# Chat with your PDF — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload any PDF and ask questions about it in plain English, with answers grounded in the document's actual content and page-level citations.

**[Try it live -> YOUR_STREAMLIT_URL_HERE]**

## What it does

- Upload any PDF
- Ask questions in natural language
- Answers are generated from the document's actual content, with source page citations
- Handles corrupted files, scanned-image PDFs, and API rate limits gracefully

## Tech stack

LangChain, Google Gemini API (LLM + embeddings), ChromaDB, Streamlit

## Run it locally

```bash
git clone https://github.com/Mehfooz-A/pdf-rag-chatbot.git
cd pdf-rag-chatbot
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_key_here
```

```bash
streamlit run app.py
```

## Deep dive

For a full explanation of the architecture, design decisions, and how RAG works under the hood, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## Notes

This is Module 1 of a larger project - an HR Intelligence Suite combining document Q&A, attrition prediction, and BI reporting for SMEs.
