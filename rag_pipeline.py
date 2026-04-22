import os
import time
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

# ── Persist DB next to this file so we never re-embed unnecessarily ────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_CHROMA_PATH = os.path.join(_DIR, ".chroma_db")
_KB_PATH = os.path.join(_DIR, "knowledge_base.md")


def _embed_with_retry(embeddings, texts, max_retries: int = 6):
    """Embed a list of texts with exponential back-off on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return embeddings.embed_documents(texts)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait = 2 ** attempt * 5   # 5 s, 10 s, 20 s, 40 s …
                print(f"⚠️  Rate-limit hit. Retrying in {wait}s … (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Embedding failed after maximum retries – please wait a minute and try again.")


def get_retriever():
    """
    Returns a LangChain retriever backed by a persistent Chroma vector store.
    The store is built once from knowledge_base.md and reused on every call.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # ── Re-use existing store if it already exists ─────────────────────────
    if os.path.exists(_CHROMA_PATH) and os.listdir(_CHROMA_PATH):
        print("✅ Loading existing vector store from disk …")
        vectorstore = Chroma(
            persist_directory=_CHROMA_PATH,
            embedding_function=embeddings,
        )
        return vectorstore.as_retriever(search_kwargs={"k": 2})

    # ── First run: build the store ─────────────────────────────────────────
    print("🔨 Building vector store for the first time …")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    with open(_KB_PATH, "r") as f:
        md_text = f.read()
    docs = splitter.split_text(md_text)

    # Embed with back-off, then save to disk
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    print(f"   Embedding {len(texts)} chunk(s) …")
    vectors = _embed_with_retry(embeddings, texts)

    vectorstore = Chroma(
        persist_directory=_CHROMA_PATH,
        embedding_function=embeddings,
    )
    vectorstore._collection.add(
        ids=[str(i) for i in range(len(texts))],
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )
    print("✅ Vector store built and saved.")

    return vectorstore.as_retriever(search_kwargs={"k": 2})


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    retriever = get_retriever()
    docs = retriever.invoke("What is the price of the pro plan?")
    print([doc.page_content for doc in docs])
