"use client";

import { useState, useRef } from "react";
import {
  Building2, Upload, X, Loader2, AlertTriangle,
  TrendingUp, TrendingDown, ScanSearch, FileText,
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────────────────────
interface Transaction {
  date: string | null;
  description: string | null;
  debit: number | null;
  credit: number | null;
  balance: number | null;
}

interface BankData {
  bank_name: string | null;
  account_holder: string | null;
  account_number: string | null;
  account_type: string | null;
  ifsc_code: string | null;
  statement_period_from: string | null;
  statement_period_to: string | null;
  opening_balance: number | null;
  closing_balance: number | null;
  transaction_count: number | null;
  total_debits: number | null;
  total_credits: number | null;
  largest_debit: number | null;
  largest_credit: number | null;
  transactions: Transaction[];
}

interface BankResponse {
  status: string;
  message: string;
  confidence_score: number;
  extracted_data: BankData;
  ocr_metadata: {
    engine_used: string;
    confidence: number;
    processing_time_ms: number;
    pages_processed: number;
  };
  processing_time_seconds: number;
}

interface DetectedRegion {
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
}

interface BankYoloResult {
  regions: DetectedRegion[];
  annotated_image_base64: string | null;
  extracted_text_per_region: Record<string, string>;
  bank_name: string | null;
  account_number: string | null;
  transactions_found: number;
  processing_time_ms: number;
}

// ── State shapes lifted to parent ────────────────────────────────────────────
interface ExtractState {
  file: File | null;
  result: BankResponse | null;
  error: string | null;
  useEnhanced: boolean;
}

interface YoloState {
  file: File | null;
  preview: string | null;
  result: BankYoloResult | null;
  error: string | null;
}

// ── Metric card ──────────────────────────────────────────────────────────────
function Metric({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="bg-surface-secondary border border-border rounded-lg p-4 relative overflow-hidden">
      <div className={`absolute top-0 left-0 w-[3px] h-full ${accent}`} />
      <p className="text-[11px] font-medium text-txt-muted uppercase tracking-wider">{label}</p>
      <p className="text-lg font-bold mt-1 text-txt-primary truncate">{value}</p>
    </div>
  );
}

// ── Extract tab ───────────────────────────────────────────────────────────────
// All result/file state lives in the PARENT — switching tabs never clears it.
function BankExtractTab({
  state,
  setState,
}: {
  state: ExtractState;
  setState: React.Dispatch<React.SetStateAction<ExtractState>>;
}) {
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const { file, result, error, useEnhanced } = state;
  const ed = result?.extracted_data;

  const handleExtract = async () => {
    if (!file) return;
    setLoading(true);
    setState(s => ({ ...s, error: null }));
    try {
      const fd = new FormData();
      fd.append("file", file);
      const endpoint = useEnhanced
        ? "/api/v2/bank/extract-enhanced"
        : "/api/v2/bank/extract";
      const r = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body: fd });
      if (r.ok) {
        const data: BankResponse = await r.json();
        setState(s => ({ ...s, result: data }));         
        } else {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        setState(s => ({ ...s, error: err.detail || `HTTP ${r.status}` }));
      }
    } catch (e: any) {
      setState(s => ({ ...s, error: e.message || "Network error" }));
    }
    setLoading(false);
  };

  return (
    <div className="space-y-5">
      <div className="bg-surface-card border border-border rounded-xl p-6">
        <h2 className="font-semibold mb-4 flex items-center gap-2 text-txt-primary">
          <FileText className="w-5 h-5 text-emerald-400" /> Upload Bank Statement
        </h2>

        <div
          className="border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all hover:border-emerald-500/50 hover:bg-surface-card/50 border-border"
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.jpg,.jpeg,.png"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) setState(s => ({ ...s, file: f, result: null, error: null }));
            }}
          />
          <Upload className="w-10 h-10 mx-auto mb-3 text-txt-muted" />
          <p className="font-semibold text-txt-primary">Drop bank statement here or click to browse</p>
          <p className="text-sm text-txt-muted mt-1">PDF, JPG, PNG — SBI, HDFC, ICICI, Axis, Kotak and 10+ banks</p>
        </div>

        {file && (
          <div className="flex items-center gap-3 mt-3 px-4 py-3 bg-surface-secondary rounded-lg border border-border">
            <FileText className="w-5 h-5 text-emerald-400 flex-shrink-0" />
            <span className="font-medium text-sm flex-1 truncate">{file.name}</span>
            <span className="text-xs text-txt-muted">{(file.size / 1024).toFixed(0)} KB</span>
            <button
              onClick={() => setState(s => ({ ...s, file: null, result: null }))}
              className="text-red-400 p-1"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        <label className="flex items-center gap-2 text-sm text-txt-secondary cursor-pointer mt-3">
          <input
            type="checkbox"
            checked={useEnhanced}
            onChange={e => setState(s => ({ ...s, useEnhanced: e.target.checked }))}
            className="w-4 h-4 accent-emerald-500"
          />
          Use YOLO-crop pipeline (better for scanned images)
        </label>

        <button
          onClick={handleExtract}
          disabled={!file || loading}
          className="w-full mt-4 flex items-center justify-center gap-2 px-5 py-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-emerald-600/40 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
        >
          {loading
            ? <><Loader2 className="w-4 h-4 animate-spin" /> Extracting…</>
            : <><Building2 className="w-4 h-4" /> Extract Statement</>}
        </button>

        {error && (
          <div className="mt-4 flex items-center gap-2 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        )}
      </div>

      {/* Results — always rendered while result exists in parent state */}
      {ed && (
        <div className="space-y-5">
          <div className="flex items-center gap-3 flex-wrap text-sm">
            <span className="px-3 py-1.5 rounded-full text-xs font-semibold bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
              {result!.status.toUpperCase()}
            </span>
            <span className="text-txt-secondary">
              Confidence: <strong>{(result!.confidence_score * 100).toFixed(0)}%</strong>
            </span>
            <span className="text-txt-secondary">
              Time: <strong>{result!.processing_time_seconds.toFixed(2)}s</strong>
            </span>
            <span className="text-xs px-2 py-0.5 bg-surface-secondary border border-border rounded text-txt-muted">
              {result!.ocr_metadata.engine_used}
            </span>
            <button
              onClick={async () => {
                try {
                  const r = await fetch(`${API_BASE}/api/v2/bank/export-xlsx`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(result!.extracted_data),
                  });
                  if (!r.ok) throw new Error(`HTTP ${r.status}`);
                  const blob = await r.blob();
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `bank_statement_${Date.now()}.xlsx`;
                  document.body.appendChild(a); a.click();
                  document.body.removeChild(a);
                  URL.revokeObjectURL(url);
                } catch (e: any) { alert(`Export failed: ${e.message}`); }
              }}
              className="ml-auto flex items-center gap-2 px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-lg transition-colors"
            >
              ⬇ Export Excel
            </button>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <Metric label="Bank"         value={ed.bank_name || "Unknown"} accent="bg-blue-500" />
            <Metric label="Transactions" value={String(ed.transaction_count ?? 0)} accent="bg-purple-500" />
            <Metric label="Opening Bal"  value={ed.opening_balance != null ? `₹${ed.opening_balance.toLocaleString()}` : "N/A"} accent="bg-amber-500" />
            <Metric label="Closing Bal"  value={ed.closing_balance != null ? `₹${ed.closing_balance.toLocaleString()}` : "N/A"} accent="bg-emerald-500" />
          </div>

          <div className="bg-surface-card border border-border rounded-xl p-5">
            <h3 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-4">Account Details</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
              {[
                ["Account Holder", ed.account_holder],
                ["Account Number", ed.account_number],
                ["Account Type",   ed.account_type],
                ["IFSC Code",      ed.ifsc_code],
                ["Period From",    ed.statement_period_from],
                ["Period To",      ed.statement_period_to],
              ].map(([label, val]) => val ? (
                <div key={label as string} className="flex justify-between py-2 border-b border-border/30 text-sm">
                  <span className="text-txt-secondary">{label}</span>
                  <span className="font-medium text-txt-primary">{val}</span>
                </div>
              ) : null)}
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <TrendingDown className="w-4 h-4 text-red-400" />
                <p className="text-xs text-txt-muted">Total Debits</p>
              </div>
              <p className="text-lg font-bold text-red-400">
                {ed.total_debits != null ? `₹${ed.total_debits.toLocaleString()}` : "—"}
              </p>
            </div>
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-1">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                <p className="text-xs text-txt-muted">Total Credits</p>
              </div>
              <p className="text-lg font-bold text-emerald-400">
                {ed.total_credits != null ? `₹${ed.total_credits.toLocaleString()}` : "—"}
              </p>
            </div>
            <div className="bg-surface-card border border-border rounded-lg p-4">
              <p className="text-xs text-txt-muted mb-1">Largest Debit</p>
              <p className="text-lg font-bold text-txt-primary">
                {ed.largest_debit != null ? `₹${ed.largest_debit.toLocaleString()}` : "—"}
              </p>
            </div>
            <div className="bg-surface-card border border-border rounded-lg p-4">
              <p className="text-xs text-txt-muted mb-1">Largest Credit</p>
              <p className="text-lg font-bold text-txt-primary">
                {ed.largest_credit != null ? `₹${ed.largest_credit.toLocaleString()}` : "—"}
              </p>
            </div>
          </div>

          {ed.transactions && ed.transactions.length > 0 && (
            <div className="bg-surface-card border border-border rounded-xl overflow-hidden">
              <div className="px-5 py-3 border-b border-border bg-surface-secondary">
                <h3 className="font-semibold text-txt-primary">
                  Transactions <span className="text-blue-400">({ed.transactions.length})</span>
                </h3>
              </div>
              <div className="overflow-x-auto max-h-96 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-surface-secondary">
                    <tr>
                      {["Date", "Description", "Debit", "Credit", "Balance"].map(h => (
                        <th key={h} className="px-4 py-2 text-left text-[11px] font-semibold text-txt-muted uppercase tracking-wider">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {ed.transactions.map((tx, i) => (
                      <tr key={i} className="border-t border-border/40 hover:bg-surface-hover transition-colors">
                        <td className="px-4 py-2 text-txt-muted whitespace-nowrap">{tx.date || "—"}</td>
                        <td className="px-4 py-2 text-txt-primary max-w-[300px] truncate">{tx.description || "—"}</td>
                        <td className="px-4 py-2 text-red-400">{tx.debit != null ? `₹${tx.debit.toLocaleString()}` : ""}</td>
                        <td className="px-4 py-2 text-emerald-400">{tx.credit != null ? `₹${tx.credit.toLocaleString()}` : ""}</td>
                        <td className="px-4 py-2 text-txt-secondary">{tx.balance != null ? `₹${tx.balance.toLocaleString()}` : ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <details className="border border-border rounded-xl overflow-hidden">
            <summary className="px-4 py-3 bg-surface-secondary cursor-pointer text-sm font-medium hover:bg-surface-hover text-txt-primary">
              📋 Full Extracted Data (JSON)
            </summary>
            <pre className="p-4 text-xs font-mono text-txt-secondary whitespace-pre-wrap break-words max-h-72 overflow-y-auto bg-surface-input">
              {JSON.stringify(result!.extracted_data, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
}

// ── YOLO tab ──────────────────────────────────────────────────────────────────
// Same pattern — state lives in parent, passed down as props.
function BankYoloTab({
  state,
  setState,
}: {
  state: YoloState;
  setState: React.Dispatch<React.SetStateAction<YoloState>>;
}) {
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const { file, preview, result, error } = state;

  const handleFile = (f: File) =>
    setState(s => ({
      ...s,
      file: f,
      preview: f.type === "application/pdf" ? null : URL.createObjectURL(f),
      result: null,
      error: null,
    }));

  // cleaner version — avoid double fetch
  const runDetect = async () => {
    if (!file) return;
    setLoading(true);
    setState(s => ({ ...s, error: null }));
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API_BASE}/api/v2/yolo/bank-detect`, {
        method: "POST", body: fd,
      });
      if (r.ok) {
        const data: BankYoloResult = await r.json();
        setState(s => ({ ...s, result: data }));
      } else {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        setState(s => ({ ...s, error: err.detail || `HTTP ${r.status}` }));
      }
    } catch (e: any) {
      setState(s => ({ ...s, error: e.message || "Network error" }));
    }
    setLoading(false);
  };

  return (
    <div className="space-y-5">
      <div className="px-4 py-3 bg-blue-500/10 border border-blue-500/20 rounded-lg text-sm text-blue-400">
        <strong>YOLO Bank Detection:</strong> Upload a photo or scan of a bank statement.
        YOLO detects header, transaction table, and footer separately — each region OCR'd individually.
      </div>

      <div className="bg-surface-card border border-border rounded-xl p-6">
        <h2 className="font-semibold mb-4 flex items-center gap-2 text-txt-primary">
          <ScanSearch className="w-5 h-5 text-purple-400" /> Bank Statement — YOLO Region Detection
        </h2>

        {!file ? (
          <div
            className="border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all hover:border-purple-500/50 border-border"
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,.pdf"
              className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
            />
            <ScanSearch className="w-10 h-10 mx-auto mb-3 text-txt-muted" />
            <p className="font-semibold text-txt-primary">Drop bank statement image here</p>
            <p className="text-sm text-txt-muted mt-1">JPG, PNG, PDF — photo, scan, or multi-page PDF</p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-3 px-4 py-3 bg-surface-secondary rounded-lg border border-border">
              <ScanSearch className="w-5 h-5 text-purple-400" />
              <span className="font-medium text-sm flex-1 truncate">{file.name}</span>
              <button
                onClick={() => setState({ file: null, preview: null, result: null, error: null })}
                className="text-red-400 p-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            {preview ? (
  <div className="rounded-xl overflow-hidden border border-border">
    <img src={preview} alt="preview" className="max-h-72 w-full object-contain bg-surface-secondary" />
  </div>
) : file?.type === "application/pdf" ? (
  <div className="flex items-center justify-center h-24 bg-surface-secondary rounded-xl border border-border text-txt-muted text-sm gap-2">
    <FileText className="w-5 h-5 text-purple-400" />
    <span>PDF ready — no preview available</span>
  </div>
) : null}
          </div>
        )}

        <button
          onClick={runDetect}
          disabled={!file || loading}
          className="w-full mt-4 flex items-center justify-center gap-2 px-5 py-3 bg-purple-600 hover:bg-purple-500 disabled:bg-purple-600/40 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
        >
          {loading
            ? <><Loader2 className="w-4 h-4 animate-spin" /> Detecting…</>
            : <><ScanSearch className="w-4 h-4" /> Run YOLO Detection</>}
        </button>

        {error && (
          <div className="mt-4 flex items-center gap-2 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400">
            <AlertTriangle className="w-4 h-4" /> {error}
          </div>
        )}
      </div>

      {result && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <Metric label="Regions found"  value={String(result.total_regions)}      accent="bg-purple-500" />
            <Metric label="Bank detected"  value={result.bank_name || "Unknown"}     accent="bg-blue-500" />
            <Metric label="Transactions"   value={String(result.transactions_found)} accent="bg-emerald-500" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div className="bg-surface-card border border-border rounded-xl p-4">
              <h3 className="text-sm font-semibold mb-3 text-txt-primary">Annotated output</h3>
              {result.annotated_image_base64 ? (
                <img
                  src={`data:image/jpeg;base64,${result.annotated_image_base64}`}
                  alt="annotated"
                  className="w-full rounded-lg border border-border"
                />
              ) : (
                <div className="flex items-center justify-center h-48 bg-surface-secondary rounded-lg text-txt-muted text-sm">
                  No annotated image
                </div>
              )}
            </div>

            <div className="bg-surface-card border border-border rounded-xl p-4 space-y-3">
              <h3 className="text-sm font-semibold text-txt-primary">OCR text per region</h3>
              {Object.entries(result.extracted_text_per_region).map(([label, text]) => (
                <div key={label} className="border border-border rounded-lg overflow-hidden">
                  <div className="px-3 py-2 bg-surface-secondary text-xs font-semibold text-purple-400 uppercase">{label}</div>
                  <pre className="p-3 text-xs text-txt-secondary whitespace-pre-wrap break-words max-h-32 overflow-y-auto bg-surface-input font-mono">
                    {text || "(empty)"}
                  </pre>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-surface-card border border-border rounded-xl p-4">
            <h3 className="text-sm font-semibold mb-3 text-txt-primary">Detected regions</h3>
            <div className="space-y-2">
              {result.regions.map((reg, i) => (
                <div key={i} className="flex items-center justify-between px-4 py-2 bg-surface-secondary rounded-lg border border-border">
                  <span className="font-medium text-sm capitalize">{reg.label}</span>
                  <div className="flex items-center gap-4">
                    <span className="text-xs font-mono text-txt-muted">
                      [{reg.bbox.map(v => Math.round(v)).join(", ")}]
                    </span>
                    <span className={`text-xs font-semibold ${
                      reg.confidence >= 0.75 ? "text-emerald-400"
                      : reg.confidence >= 0.60 ? "text-amber-400"
                      : "text-red-400"
                    }`}>
                      {(reg.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main exported panel — owns ALL state ─────────────────────────────────────
export default function BankPanel() {
  const [subTab, setSubTab] = useState<"extract" | "yolo">("extract");

  // ── KEY FIX: both state objects live HERE in the parent ──────────────────
  // Child components receive state as props and update via setState callbacks.
  // Switching tabs only changes `subTab` — neither child unmounts,
  // so all extracted results, uploaded files, and errors are preserved.
  const [extractState, setExtractState] = useState<ExtractState>({
    file: null,
    result: null,
    error: null,
    useEnhanced: false,
  });

  const [yoloState, setYoloState] = useState<YoloState>({
    file: null,
    preview: null,
    result: null,
    error: null,
  });

  return (
    <div className="space-y-5">
      {/* Sub-tabs */}
      <div className="flex gap-1 p-1 bg-surface-secondary rounded-xl border border-border">
        <button
          onClick={() => setSubTab("extract")}
          className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg text-sm font-medium transition-all ${
            subTab === "extract"
              ? "bg-emerald-600 text-white shadow-lg"
              : "text-txt-secondary hover:text-txt-primary hover:bg-surface-card"
          }`}
        >
          <FileText className="w-4 h-4" /> Extract (OCR)
        </button>
        <button
          onClick={() => setSubTab("yolo")}
          className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg text-sm font-medium transition-all ${
            subTab === "yolo"
              ? "bg-purple-600 text-white shadow-lg"
              : "text-txt-secondary hover:text-txt-primary hover:bg-surface-card"
          }`}
        >
          <ScanSearch className="w-4 h-4" /> YOLO Detection
        </button>
      </div>

      {/*
        ── KEY FIX: show/hide with CSS, NOT conditional rendering ──────────
        {subTab === "extract" && <BankExtractTab />}  ← WRONG: unmounts on switch
        
        Using className="hidden" keeps both components mounted in the DOM.
        The child's local `loading` state resets on switch (that's fine),
        but all results from parent state stay intact.
      */}
      <div className={subTab === "extract" ? "block" : "hidden"}>
        <BankExtractTab state={extractState} setState={setExtractState} />
      </div>
      <div className={subTab === "yolo" ? "block" : "hidden"}>
        <BankYoloTab state={yoloState} setState={setYoloState} />
      </div>
    </div>
  );
}