"use client";

import { useState, useRef } from "react";
import { ScanSearch, Upload, X, Loader2, AlertTriangle, Info } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────
interface DetectedRegion {
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // x1, y1, x2, y2
}

interface YoloResult {
  regions: DetectedRegion[];
  annotated_image_base64: string | null; // backend returns annotated image
  model_used: string;
  processing_time_ms: number;
  total_regions: number;
}

// Color map per region label
const REGION_COLORS: Record<string, string> = {
  header:    "bg-blue-500/10 border-blue-500/30 text-blue-400",
  table:     "bg-purple-500/10 border-purple-500/30 text-purple-400",
  footer:    "bg-amber-500/10 border-amber-500/30 text-amber-400",
  logo:      "bg-pink-500/10 border-pink-500/30 text-pink-400",
  stamp:     "bg-red-500/10 border-red-500/30 text-red-400",
  signature: "bg-teal-500/10 border-teal-500/30 text-teal-400",
  barcode:   "bg-green-500/10 border-green-500/30 text-green-400",
  address:   "bg-orange-500/10 border-orange-500/30 text-orange-400",
};

function regionColor(label: string) {
  return REGION_COLORS[label.toLowerCase()] ?? "bg-surface-secondary border-border text-txt-secondary";
}

function ConfBar({ value }: { value: number }) {
  const pct = (value * 100).toFixed(1);
  const color =
    value >= 0.85
      ? "bg-emerald-500"
      : value >= 0.65
      ? "bg-amber-500"
      : "bg-red-500";
  return (
    <div className="flex items-center gap-2 text-xs">
      <div className="flex-1 h-1.5 bg-surface-secondary rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-txt-muted w-10 text-right">{pct}%</span>
    </div>
  );
}

// ── What images work best ─────────────────────────────────────────────────
function ImageGuide() {
  const good = [
    "Scanned invoice (JPG/PNG) — physical paper invoices scanned as image",
    "Photo of invoice taken by phone — Swiggy/Zomato delivery receipt",
    "Rotated or skewed invoice — YOLO handles tilt up to ~30°",
    "Multi-column invoice — header + item table + footer visible",
    "Low contrast / faded invoice — YOLO detects regions before OCR",
    "Receipt with stamp + signature — YOLO isolates those blocks",
  ];
  const bad = [
    "Pure PDF with embedded text — use Extract tab instead (OCR is faster)",
    "Blank white image or completely black image",
    "Image smaller than 100×100px — too low resolution",
    "Non-invoice images (selfies, screenshots, etc.)",
  ];
  return (
    <div className="bg-surface-card border border-border rounded-xl p-5 flex items-start gap-4">
      <Info className="w-5 h-5 text-blue-400 flex-shrink-0" />
    </div> 
  );
}

// ── Main component ────────────────────────────────────────────────────────
export default function YoloPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<YoloResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const clearAll = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const handleDetect = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API_BASE}/api/v2/yolo/detect`, {
        method: "POST",
        body: fd,
      });
      if (r.ok) {
        setResult(await r.json());
      } else {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        setError(err.detail || `HTTP ${r.status}`);
      }
    } catch (e: any) {
      setError(e.message || "Network error — is the backend running?");
    }
    setLoading(false);
  };

  return (
    <div className="space-y-5">
      <ImageGuide />

      {/* Upload + detect */}
      <div className="bg-surface-card border border-border rounded-xl p-6">
        <h2 className="text-lg font-semibold flex items-center gap-2 mb-1">
          <ScanSearch className="w-5 h-5 text-purple-400" /> YOLO Document
          Region Detection
        </h2>
        <p className="text-sm text-txt-secondary mb-5">
            Upload an invoice image or PDF. YOLOv8 detects visual regions (header,
            item table, footer, logo, stamp, signature, barcode) and draws bounding
            boxes. PDFs are rendered at 2× zoom (first page only).
        </p>

        {/* Drop zone */}
        {!file ? (
          <div
            className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-200
              ${dragOver ? "border-purple-500 bg-purple-500/10" : "border-border hover:border-purple-500/50"}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const f = e.dataTransfer.files[0];
              if (f) handleFile(f);
            }}
            onClick={() => inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,.bmp,.tiff,.pdf"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
              }}
            />
            <Upload className="w-10 h-10 mx-auto mb-3 text-txt-muted" />
            <p className="font-semibold text-txt-primary">
              Drop invoice image here or click to browse
            </p>
            <p className="text-sm text-txt-muted mt-1">
              JPG, PNG, WebP, BMP, TIFF, PDF — Max 20MB
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* File info */}
            <div className="flex items-center gap-3 px-4 py-3 bg-surface-secondary rounded-lg border border-border">
              <ScanSearch className="w-5 h-5 text-purple-400 flex-shrink-0" />
              <span className="font-medium text-sm flex-1 truncate">
                {file.name}
              </span>
              <span className="text-xs text-txt-muted">
                {(file.size / 1024).toFixed(0)} KB
              </span>
              <button
                onClick={clearAll}
                className="text-red-400 hover:text-red-300 p-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Image preview */}
            {preview && (
  <div className="rounded-xl overflow-hidden border border-border bg-surface-secondary">
    {file?.type === "application/pdf" ? (
      <div className="flex items-center justify-center gap-3 h-32 text-txt-secondary text-sm">
        <ScanSearch className="w-6 h-6 text-purple-400" />
        <span>PDF ready — region preview will appear after detection</span>
      </div>
    ) : (
      <img
        src={preview}
        alt="invoice preview"
        className="max-h-80 w-full object-contain"
      />
    )}
  </div>
)}
          </div>
        )}

        <button
          onClick={handleDetect}
          disabled={!file || loading}
          className="w-full mt-5 flex items-center justify-center gap-2 px-5 py-3 bg-purple-600 hover:bg-purple-500 disabled:bg-purple-600/40 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" /> Detecting regions…
            </>
          ) : (
            <>
              <ScanSearch className="w-4 h-4" /> Run YOLO Detection
            </>
          )}
        </button>

        {error && (
          <div className="mt-4 flex items-start gap-2 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400">
            <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <div>
              <p>{error}</p>
              <p className="text-xs mt-1 text-red-300">
                Make sure backend is running:{" "}
                <code className="font-mono">
                  uvicorn app.main:app --reload --port 8000
                </code>
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-5 animate-slide-up">
          {/* Stats bar */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-surface-card border border-border rounded-lg p-4 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-[3px] h-full bg-purple-500" />
              <p className="text-[11px] font-medium text-txt-muted uppercase tracking-wider">
                Regions found
              </p>
              <p className="text-xl font-bold mt-1 text-txt-primary">
                {result.total_regions}
              </p>
            </div>
            <div className="bg-surface-card border border-border rounded-lg p-4 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-[3px] h-full bg-blue-500" />
              <p className="text-[11px] font-medium text-txt-muted uppercase tracking-wider">
                Model
              </p>
              <p className="text-xl font-bold mt-1 text-txt-primary">
                {result.model_used}
              </p>
            </div>
            <div className="bg-surface-card border border-border rounded-lg p-4 relative overflow-hidden">
              <div className="absolute top-0 left-0 w-[3px] h-full bg-emerald-500" />
              <p className="text-[11px] font-medium text-txt-muted uppercase tracking-wider">
                Time
              </p>
              <p className="text-xl font-bold mt-1 text-txt-primary">
                {result.processing_time_ms.toFixed(0)}ms
              </p>
            </div>
          </div>

          {/* Annotated image + region list */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {/* Annotated image */}
            <div className="bg-surface-card border border-border rounded-xl p-4">
              <h3 className="text-sm font-semibold mb-3 text-txt-primary">
                Annotated output
              </h3>
              {result.annotated_image_base64 ? (
                <img
                  src={`data:image/jpeg;base64,${result.annotated_image_base64}`}
                  alt="YOLO annotated"
                  className="w-full rounded-lg border border-border"
                />
              ) : (
                <div className="flex items-center justify-center h-48 bg-surface-secondary rounded-lg text-txt-muted text-sm">
                  No annotated image returned from backend
                </div>
              )}
            </div>

            {/* Region details */}
            <div className="bg-surface-card border border-border rounded-xl p-4">
              <h3 className="text-sm font-semibold mb-3 text-txt-primary">
                Detected regions
              </h3>
              {result.regions.length === 0 ? (
                <div className="flex items-center justify-center h-48 text-txt-muted text-sm">
                  No regions detected — try a clearer invoice image
                </div>
              ) : (
                <div className="space-y-3 max-h-96 overflow-y-auto pr-1">
                  {result.regions.map((reg, i) => (
                    <div
                      key={i}
                      className={`border rounded-lg px-4 py-3 ${regionColor(reg.label)}`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="font-semibold text-sm capitalize">
                          {reg.label}
                        </span>
                        <span className="text-xs font-mono">
                          {(reg.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <ConfBar value={reg.confidence} />
                      <p className="text-xs mt-1.5 font-mono opacity-60">
                        bbox: [{reg.bbox.map((v) => Math.round(v)).join(", ")}]
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Raw JSON */}
          <details className="border border-border rounded-xl overflow-hidden">
            <summary className="px-4 py-3 bg-surface-secondary cursor-pointer text-sm font-medium hover:bg-surface-hover">
              📋 Raw JSON response
            </summary>
            <pre className="p-4 text-xs font-mono text-txt-secondary whitespace-pre-wrap break-words max-h-64 overflow-y-auto bg-surface-input">
              {JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
}