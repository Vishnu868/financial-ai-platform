import type { ExtractionResponse, BatchStatus } from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function checkHealth(): Promise<boolean> {
  try {
    const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    return r.ok;
  } catch { return false; }
}

export async function extractInvoice(
  file: File
): Promise<{ data?: ExtractionResponse; error?: string }> {
  try {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${API_BASE}/api/v2/invoice/extract`, {
      method: "POST",
      body: fd,
    });
    if (r.ok) return { data: await r.json() };
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    return { error: err.detail || `HTTP ${r.status}` };
  } catch (e: any) {
    return { error: e.message || "Network error" };
  }
}

export async function startBatch(
  zipFile: File
): Promise<{ data?: BatchStatus; error?: string }> {
  try {
    const fd = new FormData();
    fd.append("file", zipFile);
    const r = await fetch(`${API_BASE}/api/v2/batch/process`, {
      method: "POST",
      body: fd,
    });
    if (r.ok) return { data: await r.json() };
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    return { error: err.detail || `HTTP ${r.status}` };
  } catch (e: any) {
    return { error: e.message };
  }
}

export async function getBatchStatus(
  batchId: string
): Promise<BatchStatus | null> {
  try {
    const r = await fetch(`${API_BASE}/api/v2/batch/status/${batchId}`);
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

export async function chatWithOllama(
  question: string,
  context: string
): Promise<string | null> {
  try {
    const r = await fetch("http://localhost:11434/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "mistral",
        prompt: `You are a financial document assistant. Answer using ONLY this context.\n\nCONTEXT:\n${context}\n\nQUESTION: ${question}\n\nANSWER:`,
        stream: false,
        options: { temperature: 0.1 },
      }),
      signal: AbortSignal.timeout(25000),
    });
    if (r.ok) {
      const d = await r.json();
      return d.response || null;
    }
    return null;
  } catch { return null; }
}

// ─────────────────────────────────────────────────────────────────────────────
// SINGLE INVOICE Excel download — calls /api/v2/export/xlsx
// extracted_data is the flat object (ed.seller_name, ed.total_amount, etc.)
// ─────────────────────────────────────────────────────────────────────────────

export async function downloadExcel(extractedData: any): Promise<void> {
  try {
    const r = await fetch(`${API_BASE}/api/v2/export/xlsx`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(extractedData),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `invoice_${extractedData.invoice_number || "export"}.xlsx`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (e: any) {
    alert(
      `Excel download failed: ${e.message}\nMake sure openpyxl is installed: pip install openpyxl`
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// BATCH Excel download — one sheet per invoice
// Sends array of extracted_data objects → /api/v2/export/batch-xlsx
// ─────────────────────────────────────────────────────────────────────────────

export async function downloadBatchExcel(
  results: ExtractionResponse[]
): Promise<void> {
  try {
    const r = await fetch(`${API_BASE}/api/v2/export/batch-xlsx`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(results.map((r) => r.extracted_data)),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `batch_invoices_${Date.now()}.xlsx`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (e: any) {
    // Fallback: combined CSV
    alert(
      `Excel download failed: ${e.message}\nDownloading CSV instead.`
    );
    const csv = generateCSV(results);
    downloadBlob(csv, `batch_invoices_${Date.now()}.csv`, "text/csv");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV — flat schema (all results in one file, one row per invoice)
// ─────────────────────────────────────────────────────────────────────────────

export function generateCSV(results: ExtractionResponse[]): string {
  const headers = [
    "Platform", "Invoice Number", "Order Number", "Invoice Date", "Order Date",
    "Invoice Type", "Seller Name", "Seller GST", "Seller PAN", "FSSAI",
    "Billing Address", "Shipping Address", "Billing State", "Shipping State",
    "Place of Supply", "Place of Delivery", "Reverse Charge",
    "Subtotal", "CGST Rate", "CGST Amount", "SGST Rate", "SGST Amount",
    "IGST Rate", "IGST Amount", "Total Tax",
    "Discount", "Delivery", "Packaging", "Total Amount",
    "Amount in Words", "Buyer Name", "Phone", "Payment Method",
    "Confidence %", "Validated", "Fields Extracted", "Processing Time (s)",
    "OCR Engine", "OCR Confidence %",
  ];
  const esc = (v: any) => {
    const s = String(v ?? "");
    return s.includes(",") || s.includes('"') || s.includes("\n")
      ? `"${s.replace(/"/g, '""')}"`
      : s;
  };
  const rows = results.map((r) => {
    const d = r.extracted_data;
    return [
      d.platform, d.invoice_number, d.order_number, d.invoice_date, d.order_date,
      d.invoice_type, d.seller_name, d.seller_gst, d.seller_pan, d.fssai_license,
      d.billing_address, d.shipping_address, d.billing_state_code, d.shipping_state_code,
      d.place_of_supply, d.place_of_delivery, d.reverse_charge,
      d.subtotal,
      d.cgst_rate, d.cgst_amount,
      d.sgst_rate, d.sgst_amount,
      d.igst_rate, d.igst_amount,
      d.total_tax,
      d.discount, d.delivery_charge, d.packaging_charge, d.total_amount,
      d.amount_in_words, d.buyer_name, d.buyer_phone, d.payment_method,
      `${(r.confidence_score * 100).toFixed(0)}%`,
      r.validation_passed ? "Yes" : "No",
      d.fields_extracted,
      r.processing_time_seconds.toFixed(2),
      r.ocr_metadata?.engine_used ?? "",
      r.ocr_metadata ? `${(r.ocr_metadata.confidence * 100).toFixed(1)}%` : "",
    ].map(esc).join(",");
  });
  return [headers.join(","), ...rows].join("\n");
}

export function downloadBlob(
  content: string,
  filename: string,
  type: string
) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}