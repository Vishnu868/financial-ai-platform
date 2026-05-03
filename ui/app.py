"""
Financial Document AI Platform — Complete UI.

Features:
- Single invoice upload + extraction
- Batch upload (ZIP)
- View structured extracted data
- Toggle raw OCR text
- Validation warnings display
- CSV download
- Basic RAG chatbot
- OCR metadata display

Run:
    streamlit run ui/app.py

Requires FastAPI backend running:
    uvicorn app.main:app --reload
"""

import streamlit as st
import requests
import json
import io
import time
import pandas as pd
from datetime import datetime


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Financial Document AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #4285f4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1a73e8;
        margin-bottom: 0.8rem;
    }
    .metric-label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.3rem; font-weight: 600; color: #1a1a1a; }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "all_extractions" not in st.session_state:
    st.session_state.all_extractions = []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_api_health():
    """Check if the FastAPI backend is running."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def extract_invoice(file_bytes, filename, content_type):
    """Call the invoice extraction API."""
    try:
        r = requests.post(
            f"{API_BASE}/api/v2/invoice/extract",
            files={"file": (filename, file_bytes, content_type)},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json(), None
        else:
            return None, f"API error {r.status_code}: {r.text}"
    except requests.ConnectionError:
        return None, "Cannot connect to API. Is the backend running?"
    except requests.Timeout:
        return None, "Request timed out (120s). Try a smaller file."
    except Exception as e:
        return None, str(e)


def classify_document(file_bytes, filename, content_type):
    """Call the classification API."""
    try:
        r = requests.post(
            f"{API_BASE}/api/v2/classify/document",
            files={"file": (filename, file_bytes, content_type)},
            timeout=60,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def start_batch(zip_bytes, filename):
    """Start batch processing."""
    try:
        r = requests.post(
            f"{API_BASE}/api/v2/batch/process",
            files={"file": (filename, zip_bytes, "application/zip")},
            timeout=30,
        )
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def check_batch_status(batch_id):
    """Poll batch status."""
    try:
        r = requests.get(f"{API_BASE}/api/v2/batch/status/{batch_id}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def result_to_csv(results):
    """Convert extraction results to CSV string."""
    rows = []
    for r in results:
        ed = r.get("extracted_data", {})
        seller = ed.get("seller") or {}
        buyer = ed.get("buyer") or {}
        tax = ed.get("tax") or {}
        rows.append({
            "Platform": r.get("platform", ""),
            "Invoice Number": ed.get("invoice_number", ""),
            "Order ID": ed.get("order_id", ""),
            "Date": ed.get("invoice_date", ""),
            "Seller": seller.get("name", "") if isinstance(seller, dict) else "",
            "GSTIN": seller.get("gstin", "") if isinstance(seller, dict) else "",
            "Buyer": buyer.get("name", "") if isinstance(buyer, dict) else "",
            "Subtotal": ed.get("subtotal", ""),
            "CGST": tax.get("cgst_amount", "") if isinstance(tax, dict) else "",
            "SGST": tax.get("sgst_amount", "") if isinstance(tax, dict) else "",
            "IGST": tax.get("igst_amount", "") if isinstance(tax, dict) else "",
            "Total Tax": tax.get("total_tax", "") if isinstance(tax, dict) else "",
            "Discount": ed.get("discount", ""),
            "Delivery": ed.get("delivery_charge", ""),
            "Total": ed.get("total_amount", ""),
            "Payment": ed.get("payment_method", ""),
            "Confidence": r.get("confidence_score", ""),
            "Validated": r.get("validation_passed", ""),
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def render_metric(label, value, color="#1a73e8"):
    """Render a styled metric card."""
    st.markdown(
        f'<div class="metric-card" style="border-left-color: {color};">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    api_url = st.text_input("API URL", value=API_BASE, key="api_url")
    API_BASE_OVERRIDE = api_url

    # Health check
    api_ok = check_api_health()
    if api_ok:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline — run: `uvicorn app.main:app --reload`")

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    st.write(f"Extractions: **{len(st.session_state.all_extractions)}**")

    if st.session_state.all_extractions:
        if st.button("📥 Download All as CSV", type="secondary"):
            csv = result_to_csv(st.session_state.all_extractions)
            st.download_button(
                "💾 Save CSV",
                data=csv,
                file_name=f"extractions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

    if st.button("🗑️ Clear Session"):
        st.session_state.extraction_result = None
        st.session_state.batch_results = None
        st.session_state.all_extractions = []
        st.session_state.chat_messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "**Built with:** FastAPI · PaddleOCR · EasyOCR · "
        "Mistral · LoRA · Streamlit"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown('<p class="main-header">📄 Financial Document AI Platform</p>', unsafe_allow_html=True)
st.caption("Upload invoices and bank statements → Get structured data instantly")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tab_single, tab_batch, tab_chat, tab_history = st.tabs([
    "📄 Single Upload",
    "📦 Batch Upload",
    "💬 Chat",
    "📋 History",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: SINGLE UPLOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_single:
    col_upload, col_result = st.columns([1, 2])

    with col_upload:
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader(
            "Choose an invoice or bank statement",
            type=["pdf", "jpg", "jpeg", "png", "webp"],
            key="single_upload",
            help="Supports: Amazon, Flipkart, Swiggy, Zomato, Meesho, Myntra, BigBasket, Blinkit, JioMart",
        )

        if uploaded_file:
            # Show preview for images
            if uploaded_file.type.startswith("image/"):
                st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
            else:
                st.info(f"📎 {uploaded_file.name} ({uploaded_file.size / 1024:.0f} KB)")

            if st.button("🔍 Extract Data", type="primary", use_container_width=True):
                with st.spinner("Running Ensemble OCR + Extraction + Validation..."):
                    result, error = extract_invoice(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        uploaded_file.type,
                    )

                    if error:
                        st.error(f"❌ {error}")
                    elif result:
                        st.session_state.extraction_result = result
                        st.session_state.all_extractions.append(result)
                        st.success("✅ Extraction complete!")

    with col_result:
        result = st.session_state.extraction_result
        if result:
            st.markdown("### Extraction Results")

            # Status bar
            status = result.get("status", "unknown")
            confidence = result.get("confidence_score", 0)
            validated = result.get("validation_passed", False)

            status_color = {"success": "🟢", "partial": "🟡", "low_confidence": "🔴", "failed": "🔴"}
            st.markdown(
                f"**{status_color.get(status, '⚪')} Status:** {status.upper()} | "
                f"**Confidence:** {confidence * 100:.0f}% | "
                f"**Validated:** {'✅' if validated else '⚠️'}"
            )

            # Metrics row
            ed = result.get("extracted_data", {})
            seller = ed.get("seller") or {}
            tax = ed.get("tax") or {}

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric("Platform", result.get("platform", "N/A").upper(), "#4285f4")
            with m2:
                render_metric("Total Amount", f"₹{ed.get('total_amount', 'N/A')}", "#0f9d58")
            with m3:
                render_metric("Invoice No", ed.get("invoice_number", "N/A"), "#f4b400")
            with m4:
                render_metric(
                    "Processing Time",
                    f"{result.get('processing_time_seconds', 0):.1f}s",
                    "#db4437",
                )

            # Detailed fields
            st.markdown("---")

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("#### 📋 Invoice Details")
                fields = {
                    "Order ID": ed.get("order_id"),
                    "Date": ed.get("invoice_date"),
                    "Place of Supply": ed.get("place_of_supply"),
                    "Payment Method": ed.get("payment_method"),
                }
                for k, v in fields.items():
                    if v:
                        st.write(f"**{k}:** {v}")

                st.markdown("#### 🏪 Seller")
                if isinstance(seller, dict):
                    for k, v in [
                        ("Name", seller.get("name")),
                        ("GSTIN", seller.get("gstin")),
                        ("PAN", seller.get("pan")),
                        ("State", seller.get("state")),
                    ]:
                        if v:
                            st.write(f"**{k}:** {v}")

            with detail_col2:
                st.markdown("#### 💰 Amounts")
                amounts = {
                    "Subtotal": ed.get("subtotal"),
                    "Discount": ed.get("discount"),
                    "Delivery Charge": ed.get("delivery_charge"),
                    "Packaging": ed.get("packaging_charge"),
                }
                for k, v in amounts.items():
                    if v is not None:
                        st.write(f"**{k}:** ₹{v}")

                st.markdown("#### 🧾 Tax Breakdown")
                if isinstance(tax, dict) and tax:
                    for k, v in [
                        ("CGST", tax.get("cgst_amount")),
                        ("SGST", tax.get("sgst_amount")),
                        ("IGST", tax.get("igst_amount")),
                        ("Total Tax", tax.get("total_tax")),
                    ]:
                        if v is not None:
                            st.write(f"**{k}:** ₹{v}")

                    if tax.get("is_validated"):
                        st.markdown('<div class="success-box">✅ Tax math validated</div>', unsafe_allow_html=True)

            # Validation warnings
            warnings = result.get("warnings", [])
            if warnings:
                st.markdown("---")
                st.markdown("#### ⚠️ Validation Warnings")
                for w in warnings:
                    st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

            # Expandable sections
            st.markdown("---")

            # OCR Metadata
            with st.expander("🔬 OCR Metadata"):
                ocr = result.get("ocr_metadata", {})
                oc1, oc2, oc3 = st.columns(3)
                with oc1:
                    st.metric("PaddleOCR Regions", ocr.get("paddle_regions", 0))
                with oc2:
                    st.metric("EasyOCR Regions", ocr.get("easy_regions", 0))
                with oc3:
                    st.metric("Merged Regions", ocr.get("merged_regions", 0))
                st.write(f"**Engine:** {ocr.get('engine_used', 'N/A')}")
                st.write(f"**OCR Confidence:** {ocr.get('confidence', 0) * 100:.1f}%")
                st.write(f"**LLM Fallback Used:** {'Yes' if ocr.get('fallback_used') else 'No'}")
                st.write(f"**Processing Time:** {ocr.get('processing_time_ms', 0):.0f}ms")

            # Raw OCR Text
            with st.expander("📝 Raw OCR Text"):
                raw = result.get("raw_text", "")
                if raw:
                    st.text_area("OCR Output", raw, height=300, disabled=True)
                else:
                    st.info("No raw text available")

            # Full JSON
            with st.expander("📋 Full JSON Response"):
                st.json(result)

            # Download buttons
            st.markdown("---")
            dl1, dl2 = st.columns(2)
            with dl1:
                csv = result_to_csv([result])
                st.download_button(
                    "📥 Download as CSV",
                    data=csv,
                    file_name=f"invoice_{ed.get('invoice_number', 'extracted')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    "📥 Download as JSON",
                    data=json.dumps(result, indent=2),
                    file_name=f"invoice_{ed.get('invoice_number', 'extracted')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        else:
            st.info("👆 Upload a document and click 'Extract Data' to begin")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: BATCH UPLOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_batch:
    st.markdown("### 📦 Batch Processing")
    st.caption("Upload a ZIP file containing multiple invoice images/PDFs")

    zip_file = st.file_uploader(
        "Upload ZIP archive",
        type=["zip"],
        key="batch_upload",
        help="Maximum 20 files per batch. Supports JPG, PNG, PDF inside the ZIP.",
    )

    if zip_file and st.button("🚀 Process Batch", type="primary"):
        with st.spinner("Uploading and starting batch processing..."):
            batch_info = start_batch(zip_file.getvalue(), zip_file.name)

            if batch_info:
                batch_id = batch_info.get("batch_id")
                total = batch_info.get("total_files", 0)

                st.info(f"Batch **{batch_id}** started — processing {total} files...")

                # Poll for results
                progress_bar = st.progress(0)
                status_text = st.empty()

                while True:
                    time.sleep(2)
                    status = check_batch_status(batch_id)
                    if status is None:
                        st.error("Lost connection to batch job")
                        break

                    completed = status.get("completed", 0)
                    failed = status.get("failed", 0)
                    progress = (completed + failed) / max(total, 1)
                    progress_bar.progress(min(progress, 1.0))
                    status_text.write(
                        f"Processed: {completed}/{total} | Failed: {failed}"
                    )

                    if status.get("status") == "completed":
                        st.session_state.batch_results = status
                        results = status.get("results", [])
                        st.session_state.all_extractions.extend(results)
                        st.success(
                            f"✅ Batch complete! {completed} extracted, {failed} failed"
                        )
                        break
            else:
                st.error("Failed to start batch processing")

    # Display batch results
    if st.session_state.batch_results:
        results = st.session_state.batch_results.get("results", [])
        if results:
            st.markdown("### Results")

            # Summary table
            rows = []
            for i, r in enumerate(results):
                ed = r.get("extracted_data", {})
                rows.append({
                    "#": i + 1,
                    "Platform": r.get("platform", ""),
                    "Invoice No": ed.get("invoice_number", "N/A"),
                    "Total": f"₹{ed.get('total_amount', 'N/A')}",
                    "Confidence": f"{r.get('confidence_score', 0) * 100:.0f}%",
                    "Validated": "✅" if r.get("validation_passed") else "⚠️",
                    "Time": f"{r.get('processing_time_seconds', 0):.1f}s",
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Download all
            csv = result_to_csv(results)
            st.download_button(
                "📥 Download All Results as CSV",
                data=csv,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: CHAT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_chat:
    st.markdown("### 💬 Ask Questions About Your Document")

    if not st.session_state.extraction_result:
        st.info("Upload and extract a document first (Single Upload tab), then ask questions here.")
    else:
        # Context from extracted data
        result = st.session_state.extraction_result
        ed = result.get("extracted_data", {})
        raw_text = result.get("raw_text", "")

        st.caption(
            f"Chatting about: **{ed.get('invoice_number', 'Uploaded Document')}** "
            f"({result.get('platform', 'unknown')})"
        )

        # Display chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask about the document..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Simple RAG: search raw text + extracted fields for answer
                    context = f"""
Extracted Invoice Data:
- Platform: {result.get('platform')}
- Invoice Number: {ed.get('invoice_number')}
- Order ID: {ed.get('order_id')}
- Date: {ed.get('invoice_date')}
- Seller: {(ed.get('seller') or {}).get('name')}
- GSTIN: {(ed.get('seller') or {}).get('gstin')}
- Buyer: {(ed.get('buyer') or {}).get('name')}
- Subtotal: {ed.get('subtotal')}
- Total Tax: {(ed.get('tax') or {}).get('total_tax')}
- CGST: {(ed.get('tax') or {}).get('cgst_amount')}
- SGST: {(ed.get('tax') or {}).get('sgst_amount')}
- IGST: {(ed.get('tax') or {}).get('igst_amount')}
- Discount: {ed.get('discount')}
- Delivery: {ed.get('delivery_charge')}
- Total: {ed.get('total_amount')}
- Payment: {ed.get('payment_method')}
- Validated: {result.get('validation_passed')}
- Warnings: {result.get('warnings')}

Raw OCR Text (first 1500 chars):
{raw_text[:1500]}
"""
                    # Try Ollama for answer
                    answer = None
                    try:
                        r = requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": "mistral",
                                "prompt": (
                                    f"You are a financial document assistant. "
                                    f"Answer the question using ONLY the context below. "
                                    f"Be specific with numbers.\n\n"
                                    f"CONTEXT:\n{context}\n\n"
                                    f"QUESTION: {prompt}\n\nANSWER:"
                                ),
                                "stream": False,
                                "options": {"temperature": 0.1},
                            },
                            timeout=30,
                        )
                        if r.status_code == 200:
                            answer = r.json().get("response", "")
                    except Exception:
                        pass

                    if not answer:
                        # Fallback: simple field lookup
                        q_lower = prompt.lower()
                        if "total" in q_lower and "amount" in q_lower or "grand total" in q_lower:
                            answer = f"The total amount is ₹{ed.get('total_amount', 'not found')}."
                        elif "invoice" in q_lower and "number" in q_lower:
                            answer = f"The invoice number is {ed.get('invoice_number', 'not found')}."
                        elif "gstin" in q_lower:
                            answer = f"The seller GSTIN is {(ed.get('seller') or {}).get('gstin', 'not found')}."
                        elif "seller" in q_lower or "sold by" in q_lower:
                            answer = f"The seller is {(ed.get('seller') or {}).get('name', 'not found')}."
                        elif "tax" in q_lower:
                            tax = ed.get("tax") or {}
                            answer = (
                                f"Tax breakdown: CGST ₹{tax.get('cgst_amount', 'N/A')}, "
                                f"SGST ₹{tax.get('sgst_amount', 'N/A')}, "
                                f"IGST ₹{tax.get('igst_amount', 'N/A')}, "
                                f"Total Tax ₹{tax.get('total_tax', 'N/A')}."
                            )
                        elif "date" in q_lower:
                            answer = f"The invoice date is {ed.get('invoice_date', 'not found')}."
                        elif "payment" in q_lower:
                            answer = f"Payment method: {ed.get('payment_method', 'not found')}."
                        else:
                            answer = (
                                f"Based on the extracted data, here's what I know:\n\n"
                                f"- Platform: {result.get('platform')}\n"
                                f"- Total: ₹{ed.get('total_amount')}\n"
                                f"- Date: {ed.get('invoice_date')}\n"
                                f"- Seller: {(ed.get('seller') or {}).get('name')}\n\n"
                                f"Ask me about specific fields like total, tax, GSTIN, seller, etc."
                            )

                    st.markdown(answer)
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": answer}
                    )

        # Quick question buttons
        if st.session_state.extraction_result:
            st.markdown("---")
            st.caption("Quick questions:")
            qc1, qc2, qc3 = st.columns(3)
            with qc1:
                if st.button("What is the total?", key="q_total"):
                    st.session_state.chat_messages.append(
                        {"role": "user", "content": "What is the total amount?"}
                    )
                    ed = st.session_state.extraction_result.get("extracted_data", {})
                    answer = f"The total amount is ₹{ed.get('total_amount', 'not found')}."
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.rerun()
            with qc2:
                if st.button("Show tax breakdown", key="q_tax"):
                    st.session_state.chat_messages.append(
                        {"role": "user", "content": "Show the tax breakdown"}
                    )
                    tax = (st.session_state.extraction_result.get("extracted_data", {}).get("tax") or {})
                    answer = (
                        f"Tax breakdown:\n"
                        f"- CGST: ₹{tax.get('cgst_amount', 'N/A')}\n"
                        f"- SGST: ₹{tax.get('sgst_amount', 'N/A')}\n"
                        f"- IGST: ₹{tax.get('igst_amount', 'N/A')}\n"
                        f"- Total Tax: ₹{tax.get('total_tax', 'N/A')}"
                    )
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.rerun()
            with qc3:
                if st.button("Who is the seller?", key="q_seller"):
                    st.session_state.chat_messages.append(
                        {"role": "user", "content": "Who is the seller?"}
                    )
                    seller = (st.session_state.extraction_result.get("extracted_data", {}).get("seller") or {})
                    answer = (
                        f"Seller: {seller.get('name', 'N/A')}\n"
                        f"GSTIN: {seller.get('gstin', 'N/A')}"
                    )
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: HISTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_history:
    st.markdown("### 📋 Extraction History")

    if not st.session_state.all_extractions:
        st.info("No extractions yet. Upload a document to get started.")
    else:
        rows = []
        for i, r in enumerate(st.session_state.all_extractions):
            ed = r.get("extracted_data", {})
            rows.append({
                "#": i + 1,
                "Platform": r.get("platform", ""),
                "Invoice": ed.get("invoice_number", "N/A"),
                "Total": f"₹{ed.get('total_amount', 'N/A')}",
                "Date": ed.get("invoice_date", ""),
                "Confidence": f"{r.get('confidence_score', 0) * 100:.0f}%",
                "Valid": "✅" if r.get("validation_passed") else "⚠️",
                "Status": r.get("status", ""),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        csv = result_to_csv(st.session_state.all_extractions)
        st.download_button(
            "📥 Download All as CSV",
            data=csv,
            file_name=f"all_extractions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
