"""
Financial Document AI Chatbot — Streamlit Frontend.

Run with:
    cd financial-ai-platform
    streamlit run chatbot/app.py

Make sure the FastAPI backend is running first:
    uvicorn app.main:app --reload
"""

import streamlit as st
import requests
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine import index_document, answer_question, clear_index

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="FinDoc AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM CSS — Premium dark terminal aesthetic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root tokens ── */
:root {
    --bg:       #0a0a0f;
    --bg2:      #111118;
    --bg3:      #18181f;
    --border:   #2a2a38;
    --accent:   #7c6fff;
    --accent2:  #ff6b6b;
    --accent3:  #00d4aa;
    --text:     #e8e8f0;
    --muted:    #6b6b80;
    --mono:     'Space Mono', monospace;
    --sans:     'Syne', sans-serif;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── Main container ── */
.main .block-container {
    padding: 0 2rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Top header bar ── */
.fin-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.4rem 0 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.fin-header .logo-mark {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.fin-header h1 {
    font-family: var(--sans) !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    letter-spacing: -0.02em;
    margin: 0 !important;
    padding: 0 !important;
}
.fin-header .badge {
    margin-left: auto;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--accent3);
    border: 1px solid var(--accent3);
    border-radius: 4px;
    padding: 2px 8px;
    letter-spacing: 0.08em;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}
.status-ok   { background: rgba(0,212,170,0.12); color: var(--accent3); border: 1px solid rgba(0,212,170,0.3); }
.status-warn { background: rgba(255,107,107,0.12); color: var(--accent2); border: 1px solid rgba(255,107,107,0.3); }

/* ── Metric cards row ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 1rem 0 1.5rem 0;
}
.metric-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.metric-card .mc-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-card .mc-value {
    font-family: var(--sans);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.2;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.metric-card .mc-sub {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--muted);
    margin-top: 3px;
}

/* ── Data table ── */
.field-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 0.78rem;
}
.field-table tr {
    border-bottom: 1px solid var(--border);
}
.field-table tr:last-child { border-bottom: none; }
.field-table td {
    padding: 8px 4px;
    vertical-align: top;
}
.field-table td:first-child {
    color: var(--muted);
    width: 38%;
    font-size: 0.7rem;
    letter-spacing: 0.04em;
    padding-right: 12px;
}
.field-table td:last-child {
    color: var(--text);
    font-weight: 700;
}

/* ── Section label ── */
.sec-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.2rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.3rem 0 !important;
}

[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
[data-testid="stChatMessageContent"] {
    background: transparent !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(124, 111, 255, 0.06) !important;
    border: 1px solid rgba(124, 111, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    margin: 0.4rem 0 !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    margin: 0.4rem 0 !important;
}

/* ── Chat input ── */
[data-testid="stChatInputContainer"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 2px 4px !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124, 111, 255, 0.1) !important;
}
[data-testid="stChatInput"] {
    background: transparent !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
}

/* ── Suggested question buttons ── */
.stButton > button {
    background: var(--bg3) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    padding: 8px 12px !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    background: rgba(124, 111, 255, 0.08) !important;
    color: var(--accent) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Primary button (Extract) ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), #5a4fff) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 10px 20px !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    color: var(--muted) !important;
}
.streamlit-expanderContent {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Alert / info boxes ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--accent) !important;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.04em;
}

/* ── Sidebar headers ── */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: var(--sans) !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    font-size: 0.9rem !important;
    letter-spacing: -0.01em;
}

/* ── OCR chip ── */
.ocr-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(0, 212, 170, 0.08);
    border: 1px solid rgba(0, 212, 170, 0.25);
    border-radius: 6px;
    padding: 3px 10px;
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--accent3);
    letter-spacing: 0.08em;
    margin: 2px;
}
.ocr-chip.warn {
    background: rgba(255, 107, 107, 0.08);
    border-color: rgba(255, 107, 107, 0.25);
    color: var(--accent2);
}

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem;
    text-align: center;
    gap: 12px;
}
.empty-state .es-icon {
    font-size: 3rem;
    opacity: 0.3;
}
.empty-state .es-title {
    font-family: var(--sans);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--muted);
}
.empty-state .es-sub {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    opacity: 0.6;
    max-width: 340px;
    line-height: 1.6;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: var(--border);
    border-radius: 4px;
    height: 4px;
    margin-top: 6px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent3), var(--accent));
}

/* ── Raw text box ── */
.raw-text-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    max-height: 200px;
    overflow-y: auto;
    line-height: 1.6;
    white-space: pre-wrap;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if "messages" not in st.session_state:
    st.session_state.messages = []
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "doc_indexed" not in st.session_state:
    st.session_state.doc_indexed = False
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#e8e8f0;letter-spacing:-0.02em;">◈ FinDoc AI</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#6b6b80;margin-top:2px;letter-spacing:0.08em;">DOCUMENT INTELLIGENCE v2</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Document",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Amazon, Flipkart, Swiggy, Zomato, Myntra, Meesho + Bank statements",
        label_visibility="collapsed",
    )

    api_url = st.text_input(
        "API Endpoint",
        value="http://localhost:8000",
        label_visibility="visible",
    )

    extract_btn = st.button("⬡  Extract & Index", type="primary", use_container_width=True)

    if uploaded_file and extract_btn:
        with st.spinner("Running OCR pipeline..."):
            try:
                file_payload = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }

                classify_resp = requests.post(
                    f"{api_url}/api/v2/classify/document",
                    files=file_payload, timeout=60,
                )

                if classify_resp.status_code != 200:
                    st.error(f"Classification failed: {classify_resp.text}")
                else:
                    doc_class = classify_resp.json()
                    doc_type  = doc_class.get("document_type", "unknown")
                    platform  = doc_class.get("platform", "unknown")

                    file_payload = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    endpoint = (
                        f"{api_url}/api/v2/bank/extract"
                        if doc_type == "bank_statement"
                        else f"{api_url}/api/v2/invoice/extract"
                    )

                    extract_resp = requests.post(endpoint, files=file_payload, timeout=120)

                    if extract_resp.status_code == 200:
                        result = extract_resp.json()
                        st.session_state.extracted_data = result
                        raw_text = result.get("raw_text", "")
                        st.session_state.raw_text = raw_text

                        clear_index()
                        if raw_text:
                            n_chunks = index_document(raw_text)
                            st.session_state.doc_indexed = True

                        if result.get("validation_passed"):
                            st.success("Validation passed — data consistent")
                        else:
                            for w in result.get("warnings", []):
                                st.warning(f"{w}")
                    else:
                        st.error(f"Extraction failed: {extract_resp.text}")

            except requests.ConnectionError:
                st.error("Cannot connect to API. Run: `uvicorn app.main:app --reload`")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # ── Extracted Fields Panel ──
    if st.session_state.extracted_data:
        data = st.session_state.extracted_data
        extracted = data.get("extracted_data", {})
        seller = extracted.get("seller") or {}
        conf = data.get("confidence_score", 0)
        platform_name = data.get("platform", "—").upper()
        doc_type_name = data.get("document_type", "invoice").upper()

        st.markdown('<div class="sec-label">Document</div>', unsafe_allow_html=True)

        # Status row
        validated = data.get("validation_passed", False)
        v_class = "status-ok" if validated else "status-warn"
        v_label = "VALIDATED" if validated else "WARNINGS"
        st.markdown(f"""
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;">
            <span class="status-pill status-ok">{platform_name}</span>
            <span class="status-pill {v_class}">{v_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#6b6b80;
                    letter-spacing:0.1em;margin-bottom:4px;">CONFIDENCE</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                    color:#e8e8f0;">{conf*100:.0f}%</div>
        <div class="conf-bar-wrap">
            <div class="conf-bar-fill" style="width:{conf*100:.0f}%"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-label" style="margin-top:1.2rem;">Fields</div>', unsafe_allow_html=True)

        # Field table
        invoice_no   = extracted.get("invoice_number") or "—"
        order_id     = extracted.get("order_id") or extracted.get("order_number") or "—"
        inv_date     = extracted.get("invoice_date") or "—"
        seller_name  = (seller.get("name") if isinstance(seller, dict) else None) or extracted.get("seller_name") or "—"
        gstin        = (seller.get("gstin") if isinstance(seller, dict) else None) or extracted.get("seller_gst") or "—"
        total        = extracted.get("total_amount")
        total_str    = f"₹{total:,.2f}" if total else "—"
        tax_total    = extracted.get("total_tax")
        tax_str      = f"₹{tax_total:,.2f}" if tax_total else "—"
        payment      = extracted.get("payment_method") or "—"
        supply_place = extracted.get("place_of_supply") or "—"

        rows = [
            ("INVOICE NO",   invoice_no),
            ("ORDER ID",     order_id[:20] + "…" if len(order_id) > 20 else order_id),
            ("DATE",         inv_date),
            ("SELLER",       seller_name[:22] + "…" if len(seller_name) > 22 else seller_name),
            ("GSTIN",        gstin),
            ("TOTAL",        total_str),
            ("TOTAL TAX",    tax_str),
            ("PAYMENT",      payment.upper()),
            ("SUPPLY",       supply_place),
        ]

        table_rows = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows
        )
        st.markdown(f"""
        <table class="field-table">{table_rows}</table>
        """, unsafe_allow_html=True)

        # OCR details
        ocr_meta = data.get("ocr_metadata", {})
        if ocr_meta:
            st.markdown('<div class="sec-label" style="margin-top:1rem;">OCR</div>', unsafe_allow_html=True)
            engine = ocr_meta.get("engine_used", "N/A")
            fallback = ocr_meta.get("fallback_used", False)
            proc_ms = ocr_meta.get("processing_time_ms", 0)
            st.markdown(f"""
            <div style="display:flex;flex-wrap:wrap;gap:4px;">
                <span class="ocr-chip">{engine}</span>
                <span class="ocr-chip">{proc_ms:.0f}ms</span>
                <span class="ocr-chip {'warn' if fallback else ''}">
                    {'LLM fallback' if fallback else 'No fallback'}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Full JSON"):
            st.json(data)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN AREA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Header bar
st.markdown("""
<div class="fin-header">
    <div class="logo-mark">◈</div>
    <h1>Financial Document AI</h1>
    <span class="badge">RAG · LOCAL</span>
</div>
""", unsafe_allow_html=True)

# If document is loaded — show metric cards
if st.session_state.extracted_data:
    data = st.session_state.extracted_data
    extracted = data.get("extracted_data", {})
    seller = extracted.get("seller") or {}

    total    = extracted.get("total_amount", 0) or 0
    tax      = extracted.get("total_tax", 0) or 0
    cgst     = extracted.get("cgst_amount", 0) or 0
    sgst     = extracted.get("sgst_amount", 0) or 0
    platform = data.get("platform", "—").capitalize()
    inv_no   = extracted.get("invoice_number") or "—"
    inv_date = extracted.get("invoice_date") or "—"
    n_chunks = st.session_state.get("n_chunks", "—")

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="mc-label">Total Amount</div>
            <div class="mc-value">₹{total:,.0f}</div>
            <div class="mc-sub">incl. tax</div>
        </div>
        <div class="metric-card">
            <div class="mc-label">Total Tax</div>
            <div class="mc-value">₹{tax:,.2f}</div>
            <div class="mc-sub">CGST ₹{cgst:.2f} · SGST ₹{sgst:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="mc-label">Platform</div>
            <div class="mc-value">{platform}</div>
            <div class="mc-sub">{inv_no}</div>
        </div>
        <div class="metric-card">
            <div class="mc-label">Invoice Date</div>
            <div class="mc-value">{inv_date}</div>
            <div class="mc-sub">RAG ready · {"indexed" if st.session_state.doc_indexed else "not indexed"}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Chat section label
st.markdown('<div class="sec-label">Conversation</div>', unsafe_allow_html=True)

# Empty state
if not st.session_state.messages and not st.session_state.doc_indexed:
    st.markdown("""
    <div class="empty-state">
        <div class="es-icon">◈</div>
        <div class="es-title">No document loaded</div>
        <div class="es-sub">Upload a PDF or image invoice in the sidebar,<br>then ask questions about it here.</div>
    </div>
    """, unsafe_allow_html=True)
elif not st.session_state.messages and st.session_state.doc_indexed:
    st.markdown("""
    <div class="empty-state">
        <div class="es-icon">◈</div>
        <div class="es-title">Document indexed</div>
        <div class="es-sub">Ask anything about your invoice below.<br>Try the suggested questions or type your own.</div>
    </div>
    """, unsafe_allow_html=True)

# Message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("Ask anything about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.doc_indexed:
        with st.chat_message("assistant"):
            with st.spinner("Mistral is thinking... (30-60s on CPU)"):
                result = answer_question(prompt)
                answer = result.get("answer") or "No answer returned."
                sources = result.get("sources", [])
                model_used = result.get("model", "—")

            st.markdown(answer)
            st.caption(f"Model: {model_used}")
  
            if sources:
                with st.expander(f"Sources · {model_used}"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"""
                        <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                                    color:#6b6b80;padding:6px 0;border-bottom:1px solid #2a2a38;">
                            <span style="color:#7c6fff;font-weight:700;">#{i}</span> {src[:200]}…
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        answer = "⬡ Upload and extract a document first using the sidebar."
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    if sources:
        with st.expander(f"Sources · {model_used}"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"""
                        <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                                    color:#6b6b80;padding:6px 0;border-bottom:1px solid #2a2a38;">
                            <span style="color:#7c6fff;font-weight:700;">#{i}</span> {src[:200]}…
                        </div>
                        """, unsafe_allow_html=True)
    else:
            answer = "⬡ Upload and extract a document first using the sidebar."
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Suggested Questions ──
if st.session_state.doc_indexed:
    st.markdown('<div class="sec-label" style="margin-top:1rem;">Quick questions</div>', unsafe_allow_html=True)
    questions = [
        "What is the total amount?",
        "What is the invoice number?",
        "Who is the seller?",
        "What is the GSTIN?",
        "What taxes were applied?",
        "What items were purchased?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(questions):
        with cols[i % 3]:
            if st.button(q, key=f"sq_{i}"):
                st.session_state.messages.append({"role": "user", "content": q})
                result = answer_question(q)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"]}
                )
                st.rerun()