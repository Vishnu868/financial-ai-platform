"use client";

import { useState } from "react";
import { Package, Download, Loader2, AlertTriangle, ChevronDown, ChevronUp, FileSpreadsheet } from "lucide-react";
import FileUpload from "./FileUpload";
import ResultDisplay from "./ResultDisplay";
import type { ExtractionResponse, BatchStatus } from "@/types/api";
import { startBatch, getBatchStatus, generateCSV, downloadBlob } from "@/lib/api";

interface BatchPanelProps {
  onResults: (results: ExtractionResponse[]) => void;
}

// ── Collapsible wrapper for each invoice result ──────────────────────────
function InvoiceAccordion({
  result,
  index,
}: {
  result: ExtractionResponse;
  index: number;
}) {
  const [open, setOpen] = useState(false);
  const ed = result.extracted_data;

  return (
    <div className="border border-border rounded-xl overflow-hidden">
      {/* Summary row — click to expand */}
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-4 bg-surface-secondary hover:bg-surface-hover transition-colors text-sm"
      >
        <div className="flex items-center gap-4 text-left flex-wrap">
          <span className="text-txt-muted w-6 text-right">{index + 1}</span>
          <span className="font-semibold text-txt-primary uppercase">
            {ed.platform || "UNKNOWN"}
          </span>
          <span className="text-txt-secondary">
            {ed.invoice_number || "No Invoice #"}
          </span>
          <span className="text-emerald-400 font-medium">
            ₹{ed.total_amount ?? "N/A"}
          </span>
          <span
            className={`text-xs px-2 py-0.5 rounded-full border ${
              result.validation_passed
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                : "border-amber-500/30 bg-amber-500/10 text-amber-400"
            }`}
          >
            {(result.confidence_score * 100).toFixed(0)}% conf
          </span>
          <span className="text-xs text-txt-muted">
            {ed.fields_extracted}/22 fields
          </span>
          <span className="text-xs text-txt-muted">
            {result.processing_time_seconds.toFixed(1)}s
          </span>
        </div>
        {open ? (
          <ChevronUp className="w-4 h-4 flex-shrink-0" />
        ) : (
          <ChevronDown className="w-4 h-4 flex-shrink-0" />
        )}
      </button>

      {/* Full ResultDisplay when expanded */}
      {open && (
        <div className="p-5 border-t border-border bg-surface-card animate-fade-in">
          <ResultDisplay result={result} />
        </div>
      )}
    </div>
  );
}

// ── Multi-sheet Excel download ────────────────────────────────────────────
// Each invoice → its own sheet named "Invoice_1", "Invoice_2", etc.
async function downloadBatchExcel(results: ExtractionResponse[]) {
  // Try backend first
  try {
    const API_BASE =
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const r = await fetch(`${API_BASE}/api/v2/export/batch-xlsx`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(results.map((r) => r.extracted_data)),
    });
    if (r.ok) {
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `batch_invoices_${Date.now()}.xlsx`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      return;
    }
  } catch (_) {
    // fallback to CSV if backend unavailable
  }

  // Fallback: download one CSV per invoice zipped together
  // (since we can't generate xlsx in browser without a library)
  // We'll create a combined CSV with clear section separators
  const sections: string[] = [];
  results.forEach((result, i) => {
    const ed = result.extracted_data;
    sections.push(
      `\n========== INVOICE ${i + 1}: ${ed.platform?.toUpperCase() || "UNKNOWN"} — ${ed.invoice_number || "N/A"} ==========\n`
    );
    sections.push(generateCSV([result]));
  });
  downloadBlob(
    sections.join("\n"),
    `batch_invoices_${Date.now()}.csv`,
    "text/csv"
  );
}

// ── Main component ────────────────────────────────────────────────────────
export default function BatchPanel({ onResults }: BatchPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState("");
  const [results, setResults] = useState<ExtractionResponse[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleProcess = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults(null);
    setProgress(0);

    const { data: batch, error: startErr } = await startBatch(file);
    if (startErr || !batch) {
      setError(startErr || "Failed to start batch");
      setLoading(false);
      return;
    }

    const batchId = batch.batch_id;
    const total = batch.total_files;
    setStatusText(`Processing ${total} files…`);

    while (true) {
      await new Promise((r) => setTimeout(r, 2000));
      const status = await getBatchStatus(batchId);
      if (!status) {
        setError("Lost connection to batch job");
        break;
      }

      const done = status.completed + status.failed;
      setProgress(Math.min((done / total) * 100, 100));
      setStatusText(
        `${status.completed}/${total} done${status.failed ? `, ${status.failed} failed` : ""}`
      );

      if (status.status === "completed") {
        const r = status.results || [];
        setResults(r);
        onResults(r);
        break;
      }
    }

    setLoading(false);
  };

  return (
    <div className="space-y-5">
      {/* Upload card */}
      <div className="bg-surface-card border border-border rounded-xl p-6">
        <h2 className="text-lg font-semibold flex items-center gap-2 mb-1">
          <Package className="w-5 h-5 text-blue-400" /> Batch Processing
        </h2>
        <p className="text-sm text-txt-secondary mb-5">
          Upload a ZIP file containing multiple invoice PDFs/images. Max 20
          files. Each invoice will be fully extracted with all 22+ fields.
        </p>

        <FileUpload
          accept=".zip"
          label="Drop ZIP file here or click to browse"
          hint=".zip containing invoice PDFs or images (JPG, PNG, PDF)"
          selectedFile={file}
          onFileSelect={setFile}
          onClear={() => setFile(null)}
        />

        <button
          onClick={handleProcess}
          disabled={!file || loading}
          className="w-full mt-4 flex items-center justify-center gap-2 px-5 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-600/40 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" /> Processing…
            </>
          ) : (
            <>
              <Package className="w-4 h-4" /> Process Batch
            </>
          )}
        </button>

        {loading && (
          <div className="mt-5">
            <div className="flex justify-between text-xs text-txt-muted mb-2">
              <span>{statusText}</span>
              <span>{progress.toFixed(0)}%</span>
            </div>
            <div className="w-full h-2 bg-surface-secondary rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 flex items-center gap-2 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        )}
      </div>

      {/* Results */}
      {results && results.length > 0 && (
        <div className="space-y-4 animate-slide-up">
          {/* Header + download buttons */}
          <div className="flex items-center justify-between flex-wrap gap-3 px-1">
            <h3 className="font-semibold text-txt-primary">
              Results —{" "}
              <span className="text-blue-400">{results.length} invoices</span>
            </h3>
            <div className="flex gap-2 flex-wrap">
              {/* CSV — one file, all invoices */}
              <button
                onClick={() => {
                  const csv = generateCSV(results);
                  downloadBlob(
                    csv,
                    `batch_all_${Date.now()}.csv`,
                    "text/csv"
                  );
                }}
                className="flex items-center gap-1.5 text-xs px-3 py-2 bg-surface-card border border-border rounded-lg hover:border-blue-500/40 transition-colors"
              >
                <Download className="w-3.5 h-3.5" /> Download All CSV
              </button>

              {/* Excel — one sheet per invoice */}
              <button
                onClick={() => downloadBatchExcel(results)}
                className="flex items-center gap-1.5 text-xs px-3 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition-colors"
              >
                <FileSpreadsheet className="w-3.5 h-3.5" /> Download Excel
                (separate sheets)
              </button>
            </div>
          </div>

          {/* Quick summary table */}
          <div className="bg-surface-card border border-border rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-surface-secondary text-left">
                    {[
                      "#",
                      "Platform",
                      "Invoice #",
                      "Date",
                      "Seller",
                      "Total",
                      "Fields",
                      "Conf",
                      "Valid",
                      "Time",
                    ].map((h) => (
                      <th
                        key={h}
                        className="px-4 py-3 text-[11px] font-semibold text-txt-muted uppercase tracking-wider whitespace-nowrap"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => {
                    const ed = r.extracted_data;
                    return (
                      <tr
                        key={i}
                        className="border-t border-border/40 hover:bg-surface-hover transition-colors"
                      >
                        <td className="px-4 py-3 text-txt-muted">{i + 1}</td>
                        <td className="px-4 py-3 font-medium">
                          {(ed.platform || "UNKNOWN").toUpperCase()}
                        </td>
                        <td className="px-4 py-3 font-mono text-xs">
                          {ed.invoice_number || "—"}
                        </td>
                        <td className="px-4 py-3 text-txt-secondary">
                          {ed.invoice_date || "—"}
                        </td>
                        <td className="px-4 py-3 text-txt-secondary max-w-[160px] truncate">
                          {ed.seller_name || "—"}
                        </td>
                        <td className="px-4 py-3 text-emerald-400 font-medium">
                          {ed.total_amount != null ? `₹${ed.total_amount}` : "—"}
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-xs px-2 py-0.5 bg-blue-500/10 border border-blue-500/20 rounded text-blue-400">
                            {ed.fields_extracted}/22
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          {(r.confidence_score * 100).toFixed(0)}%
                        </td>
                        <td className="px-4 py-3">
                          {r.validation_passed ? (
                            <span className="text-emerald-400">✅</span>
                          ) : (
                            <span className="text-amber-400">⚠️</span>
                          )}
                        </td>
                        <td className="px-4 py-3 text-txt-muted">
                          {r.processing_time_seconds.toFixed(1)}s
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Full detail per invoice (accordion) */}
          <div className="space-y-3">
            <p className="text-xs text-txt-muted px-1">
              Click any invoice below to see full 22-field extraction, OCR
              details, and download options:
            </p>
            {results.map((r, i) => (
              <InvoiceAccordion key={i} result={r} index={i} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}