"""
RAG (Retrieval Augmented Generation) Engine.

Flow:
1. Document text → split into chunks (400 chars, 60 overlap)
2. Chunks → embedded using sentence-transformers (all-MiniLM-L6-v2)
3. Embeddings → stored in ChromaDB (local vector store)
4. User question → embedded → find top-3 similar chunks
5. Chunks + question → sent to Mistral (via Ollama) → grounded answer

All local. No API keys. No data leaves your machine.

Compatibility: sentence-transformers 2.2.2 - 3.x, langchain-community 0.0.38+
"""

import os
import shutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Settings with fallback ───────────────────────────────────────────────────
try:
    from app.config import settings
    EMBEDDING_MODEL  = settings.embedding_model
    CHROMA_DIR       = settings.chroma_persist_dir
    CHUNK_SIZE       = settings.chunk_size
    CHUNK_OVERLAP    = settings.chunk_overlap
    RETRIEVAL_K      = settings.retrieval_k
    OLLAMA_MODEL     = settings.ollama_model
    LLM_TEMPERATURE  = settings.llm_temperature
except ImportError:
    EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # short name works for ST 2.x + 3.x
    CHROMA_DIR       = "./chroma_store"
    CHUNK_SIZE       = 400
    CHUNK_OVERLAP    = 60
    RETRIEVAL_K      = 3
    OLLAMA_MODEL     = "mistral"
    LLM_TEMPERATURE  = 0.1


# ── Embedding loader — handles ST 2.x and 3.x import paths ──────────────────
def get_embeddings():
    errors = {}

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        errors["langchain_community"] = str(e)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
    except Exception as e:
        errors["langchain_huggingface"] = str(e)

    try:
        from sentence_transformers import SentenceTransformer

        class _STEmbeddings:
            def __init__(self, model_name):
                self.model = SentenceTransformer(model_name)
            def embed_documents(self, texts):
                return self.model.encode(texts, normalize_embeddings=True).tolist()
            def embed_query(self, text):
                return self.model.encode([text], normalize_embeddings=True)[0].tolist()

        logger.info("Using bare sentence-transformers wrapper.")
        return _STEmbeddings(EMBEDDING_MODEL)

    except Exception as e:
        errors["sentence_transformers"] = str(e)

    raise RuntimeError(f"No embedding backend available: {errors}")

# ── Chroma loader — handles persist() deprecation in chromadb 0.4+ ──────────
def _get_vectorstore(embeddings, collection_name: str, from_documents=None):
    """
    Create or load a Chroma vectorstore.
    chromadb < 0.4.14 needs .persist(); newer versions auto-persist.
    """
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma

    if from_documents is not None:
        vs = Chroma.from_documents(
            documents=from_documents,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name=collection_name,
        )
        # persist() was removed in chromadb 0.4.14+ (auto-persists)
        if hasattr(vs, "persist"):
            try:
                vs.persist()
            except Exception:
                pass
        return vs

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


# ── index_document ───────────────────────────────────────────────────────────
def index_document(text: str, doc_id: str = "current_doc") -> int:
    """
    Split document text into chunks, embed, and store in ChromaDB.

    Args:
        text:   Raw OCR/PDF text from the document
        doc_id: Unique identifier for this document's collection

    Returns:
        Number of chunks created and indexed
    """
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"doc_id": doc_id}],
    )

    if not chunks:
        logger.warning("No chunks created — text may be empty.")
        return 0

    embeddings = get_embeddings()
    _get_vectorstore(embeddings, collection_name=doc_id, from_documents=chunks)

    logger.info(f"Indexed {len(chunks)} chunks for doc '{doc_id}'")
    return len(chunks)


# ── answer_question ──────────────────────────────────────────────────────────
def answer_question(
    question: str,
    doc_id: str = "current_doc",
    extracted_data: Optional[dict] = None,
) -> dict:
    """
    Answer a question using RAG: retrieve relevant chunks then generate answer.

    Args:
        question:       User's natural language question
        doc_id:         Which document collection to search
        extracted_data: Structured JSON from FastAPI (used to enrich context)

    Returns:
        Dict with keys: answer, sources, model
    """
    embeddings = get_embeddings()

    try:
        vectorstore = _get_vectorstore(embeddings, collection_name=doc_id)
    except Exception as e:
        return {
            "answer": "⬡ No document indexed yet. Upload and extract a document first.",
            "sources": [],
            "model": "none",
        }

    # Retrieve top-k relevant chunks
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K},
        )
        # get_relevant_documents deprecated in newer langchain — handle both
        if hasattr(retriever, "invoke"):
            relevant_chunks = retriever.invoke(question)
        else:
            relevant_chunks = retriever.get_relevant_documents(question)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        relevant_chunks = []

    if not relevant_chunks:
        return {
            "answer": "Could not find relevant information in the document.",
            "sources": [],
            "model": "none",
        }

    # Build context — raw OCR chunks
    context = "\n\n".join([c.page_content for c in relevant_chunks])

    # Optionally enrich with structured extracted fields
    structured_context = ""
    if extracted_data:
        ed = extracted_data.get("extracted_data", {})
        seller = ed.get("seller") or {}
        structured_context = f"""
STRUCTURED FIELDS (already extracted):
- Invoice No:    {ed.get('invoice_number') or ed.get('invoice_no', 'N/A')}
- Order No:      {ed.get('order_number') or ed.get('order_id', 'N/A')}
- Date:          {ed.get('invoice_date', 'N/A')}
- Seller:        {ed.get('seller_name') or (seller.get('name') if isinstance(seller, dict) else 'N/A')}
- GSTIN:         {ed.get('seller_gst') or (seller.get('gstin') if isinstance(seller, dict) else 'N/A')}
- Total Amount:  ₹{ed.get('total_amount', 'N/A')}
- Total Tax:     ₹{ed.get('total_tax', 'N/A')}
- CGST:          ₹{ed.get('cgst_amount', 'N/A')}
- SGST:          ₹{ed.get('sgst_amount', 'N/A')}
- Payment:       {ed.get('payment_method', 'N/A')}
- Place of Supply: {ed.get('place_of_supply', 'N/A')}
"""

    prompt = f"""You are a financial document assistant for Indian e-commerce invoices.
Use ONLY the document context and structured fields below to answer the question.
If the answer is not present, say exactly: "This information is not in the document."
Be specific — use exact numbers, dates, GST numbers, and names from the document.
Do not make up information.
{structured_context}
RAW DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    # ── LLM: try Ollama, fall back to structured context answer ──────────────
    try:
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            from langchain.llms import Ollama

        llm = Ollama(model=OLLAMA_MODEL, temperature=LLM_TEMPERATURE)
        answer = llm.invoke(prompt)
        model_used = OLLAMA_MODEL

    except Exception as e:
        logger.warning(f"Ollama unavailable ({e}). Using structured fallback.")

        # Smart fallback: try to answer from structured fields directly
        q_lower = question.lower()
        fallback_answer = _rule_based_answer(q_lower, extracted_data)
        if fallback_answer:
            answer = fallback_answer
            model_used = "structured-fields (Ollama offline)"
        else:
            answer = (
                f"**Ollama is offline** — install it from https://ollama.com and run `ollama pull mistral`\n\n"
                f"**From the document:**\n\n{context[:600]}"
            )
            model_used = "fallback (raw context)"

    return {
        "answer": answer,
        "sources": [c.page_content[:150] for c in relevant_chunks],
        "model": model_used,
    }


def _rule_based_answer(question: str, extracted_data: Optional[dict]) -> Optional[str]:
    """
    Answer common questions directly from structured extracted fields
    when Ollama is not available. Returns None if no match.
    """
    if not extracted_data:
        return None

    ed = extracted_data.get("extracted_data", {})
    seller = ed.get("seller") or {}

    def seller_name():
        return ed.get("seller_name") or (seller.get("name") if isinstance(seller, dict) else None)

    def gstin():
        return ed.get("seller_gst") or (seller.get("gstin") if isinstance(seller, dict) else None)

    rules = [
        (["total amount", "total", "how much", "grand total", "bill amount"],
         lambda: f"The total amount is **₹{ed.get('total_amount', 'N/A')}**"),

        (["invoice number", "invoice no"],
         lambda: f"The invoice number is **{ed.get('invoice_number') or ed.get('invoice_no', 'N/A')}**"),

        (["order number", "order id", "order no"],
         lambda: f"The order number is **{ed.get('order_number') or ed.get('order_id', 'N/A')}**"),

        (["date", "invoice date", "when"],
         lambda: f"The invoice date is **{ed.get('invoice_date', 'N/A')}**"),

        (["seller", "sold by", "vendor", "company"],
         lambda: f"The seller is **{seller_name() or 'N/A'}**"),

        (["gstin", "gst number", "gst no", "tax registration"],
         lambda: f"The seller's GSTIN is **{gstin() or 'N/A'}**"),

        (["cgst", "central gst"],
         lambda: f"CGST amount is **₹{ed.get('cgst_amount', 'N/A')}**"),

        (["sgst", "state gst"],
         lambda: f"SGST amount is **₹{ed.get('sgst_amount', 'N/A')}**"),

        (["tax", "taxes"],
         lambda: f"Total tax is **₹{ed.get('total_tax', 'N/A')}** "
                 f"(CGST: ₹{ed.get('cgst_amount', 0)}, SGST: ₹{ed.get('sgst_amount', 0)})"),

        (["payment", "paid", "cod", "online"],
         lambda: f"Payment method is **{str(ed.get('payment_method', 'N/A')).upper()}**"),

        (["items", "products", "purchased", "bought", "what did"],
         lambda: f"Items purchased: **{ed.get('items') or 'See raw invoice text for item details'}**"),

        (["address", "billing", "shipping"],
         lambda: f"Billing: {ed.get('billing_address', 'N/A')}\nShipping: {ed.get('shipping_address', 'N/A')}"),
    ]

    for keywords, answer_fn in rules:
        if any(kw in question for kw in keywords):
            return answer_fn()

    return None


# ── clear_index ──────────────────────────────────────────────────────────────
def clear_index(doc_id: str = "current_doc"):
    """Clear the vector store when a new document is uploaded."""
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        logger.info(f"Cleared vector store at {CHROMA_DIR}")