# Architecture & Design: PDF RAG Chatbot

This document explains, from first principles, what this system does, why each design decision was made, and how the pieces fit together. It assumes no prior familiarity with RAG systems.

---

## 1. The Problem This Solves

Large Language Models (LLMs) like Gemini are trained on a fixed snapshot of general text and have no access to a specific user's private documents. Ask an LLM a question about a document it has never seen, and it will either say it doesn't know, or worse, generate a plausible-sounding but fabricated answer (a "hallucination").

Retrieval-Augmented Generation (RAG) solves this by giving the model relevant excerpts from the actual source document at the moment it answers, rather than relying on what it memorized during training.

## 2. Why Not Just Paste the Whole Document Into the Prompt?

For a short document, this would work. It fails for two reasons:

1. Context window limits - every LLM has a maximum amount of text it can process per request. Large documents exceed this.
2. Signal dilution - even within the context window, giving a model an entire 50-page document to answer one question makes it harder to focus on the relevant part, similar to skimming an entire book to answer one question instead of being handed the right page directly.

The system instead retrieves only the most relevant small sections for each question, and gives the model only those.

## 3. How Relevance Is Determined: Embeddings

An embedding model converts text into a list of numbers (a vector), such that texts with similar meaning produce vectors that are numerically close together, regardless of shared words.

Example: "vacation policy" and "time off guidelines" share no words but are close in meaning, so a good embedding model places their vectors near each other. This lets the system match a question to relevant text even with completely different phrasing.

This project uses Google's gemini-embedding-001 model, for both the document chunks (at indexing time) and the user's question (at query time).

## 4. Why the Document Is Split Into Chunks

Embedding an entire document as one vector would average its meaning across every topic, making retrieval imprecise. Instead, the document is split into overlapping chunks (~1000 characters, 100 character overlap), each with its own embedding.

Overlap exists because splitting can cut an idea in half at a chunk boundary; a small overlap prevents context from being fully lost at that seam.

1000 characters is a middle ground: smaller chunks give more precise retrieval but risk losing needed context and increase API calls; larger chunks preserve context but reduce precision.

## 5. Where the Chunks Are Stored: ChromaDB

Once each chunk has an embedding, it needs to be stored somewhere that can answer: given a new vector, which stored vectors are closest to it? This project uses ChromaDB, an open-source vector database built for this. A fresh, in-memory ChromaDB instance is built for every new PDF upload, since each session may involve a different document.

## 6. The Retrieval Step

1. The question is embedded using the same embedding model as the chunks.
2. ChromaDB is queried for the k=3 chunks most similar to the question's embedding (cosine similarity).
3. Those chunks are formatted into a single context block.

This step involves no LLM call - it is a mathematical nearest-neighbor search, and is fast and cheap relative to generation.

## 7. The Generation Step

The retrieved context and question are combined into a prompt template instructing the model to answer only from the given context, and to say it does not know if the answer is not present. This explicit instruction is a deliberate hallucination mitigation - it gives the model a sanctioned way to decline instead of guessing.

## 8. Why LangChain

LangChain provides composable chain syntax (prompt | model | parser), standardized interfaces across LLM providers and vector stores, and a retriever abstraction that treats similarity search as just another chain step - making the pipeline easier to read, modify, and extend than wiring raw API calls by hand.

## 9. Handling API Rate Limits

Google's free tier caps embedding requests at 100 per minute. Large documents can exceed this in one burst. This system batches chunks into groups of 90, embeds one batch at a time, and pauses 65 seconds between batches when more than one batch is needed, with a visible status message so the user understands the pause. A paid tier would not need this.

## 10. Error Handling Philosophy

Unreadable PDFs (e.g. scanned images with no text layer) are detected immediately and surfaced clearly rather than proceeding into a meaningless pipeline. Any other failure during processing or answering is caught and shown as a readable message rather than a raw stack trace, keeping the app usable for a retry.

## 11. Session State and Why It Matters

Streamlit re-runs the entire script on every interaction. Without explicit state, the vector store, chain, and chat history would be rebuilt from scratch every time. st.session_state persists the built chain, the retriever, and the chat history across reruns.

## 12. Source Citations

The same retriever call used to build an answer is also used to extract each retrieved chunk's page number and a snippet, shown in a collapsible Sources section - letting the user verify the answer against its source rather than trusting it blindly, which matters more in a client-facing tool than a personal script.

## 13. Deployment and Secrets Management

The app is deployed on Streamlit Community Cloud, building directly from this GitHub repository on every push to main. The Gemini API key is never committed to the repo; it is provided via Streamlit Cloud's encrypted Secrets mechanism, functionally equivalent to a local .env file but scoped to the cloud environment.

## 14. Known Limitations

- Complex PDF layouts (multi-column, tables) can extract in a disordered order, since pypdf extracts text in document order without reconstructing visual structure.
- Free-tier rate limits mean large documents take longer to process than a paid tier would allow.
- The system answers only from retrieved context; it does not reason across multiple documents or retain memory across sessions.

## 15. Possible Extensions

- Persist the vector store per document to avoid re-embedding on re-upload.
- Add conversational memory for follow-up questions within a session.
- Support additional file types (DOCX, CSV, plain text) via additional LangChain document loaders.
