"use client";

import { useState } from "react";
import {
  CheckCircle, AlertTriangle, Shield, ChevronDown, ChevronUp,
  Download, FileSpreadsheet,
} from "lucide-react";
import type { ExtractionResponse } from "@/types/api";
import { generateCSV, downloadBlob, downloadExcel } from "@/lib/api";

interface ResultDisplayProps { result: ExtractionResponse; }

const STATUS_CFG = {
  success: { icon: CheckCircle, color: "text-emerald-400", bg: "bg-emerald-500/10", label: "SUCCESS" },
  partial: { icon: AlertTriangle, color: "text-amber-400", bg: "bg-amber-500/10", label: "PARTIAL" },
  low_confidence: { icon: AlertTriangle, color: "text-red-400", bg: "bg-red-500/10", label: "LOW CONFIDENCE" },
  failed: { icon: AlertTriangle, color: "text-red-400", bg: "bg-red-500/10", label: "FAILED" },
};

function Metric({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="bg-surface-secondary border border-border rounded-lg p-4 relative overflow-hidden">
      <div className={`absolute top-0 left-0 w-[3px] h-full ${accent}`} />
      <p className="text-[11px] font-medium text-txt-muted uppercase tracking-wider">{label}</p>
      <p className="text-xl font-bold mt-1 text-txt-primary truncate">{value}</p>
    </div>
  );
}

function FieldRow({ label, value }: { label: string; value: any }) {
  if (value === null || value === undefined || value === "") return null;
  return (
    <div className="flex justify-between py-[7px] border-b border-border/40 text-sm gap-4">
      <span className="text-txt-secondary whitespace-nowrap">{label}</span>
      <span className="font-medium text-txt-primary text-right break-words max-w-[60%]">{String(value)}</span>
    </div>
  );
}

function Collapsible({ title, children, icon }: { title: string; children: React.ReactNode; icon: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <button onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-surface-secondary hover:bg-surface-hover text-sm font-medium transition-colors">
        <span>{icon} {title}</span>
        {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
      </button>
      {open && <div className="animate-fade-in">{children}</div>}
    </div>
  );
}

// The 22 mentor fields for the grid display
const MENTOR_FIELDS: { key: string; label: string }[] = [
  { key: "billing_address", label: "billing_address" },
  { key: "shipping_address", label: "shipping_address" },
  { key: "invoice_type", label: "invoice_type" },
  { key: "order_number", label: "order_number" },
  { key: "invoice_number", label: "invoice_number" },
  { key: "order_date", label: "order_date" },
  { key: "invoice_details", label: "invoice_details" },
  { key: "invoice_date", label: "invoice_date" },
  { key: "seller_info", label: "seller_info" },
  { key: "seller_pan", label: "seller_pan" },
  { key: "seller_gst", label: "seller_gst" },
  { key: "fssai_license", label: "fssai_license" },
  { key: "billing_state_code", label: "billing_state_code" },
  { key: "shipping_state_code", label: "shipping_state_code" },
  { key: "place_of_supply", label: "place_of_supply" },
  { key: "place_of_delivery", label: "place_of_delivery" },
  { key: "reverse_charge", label: "reverse_charge" },
  { key: "amount_in_words", label: "amount_in_words" },
  { key: "seller_name", label: "seller_name" },
  { key: "seller_address", label: "seller_address" },
  { key: "total_tax", label: "total_tax" },
  { key: "total_amount", label: "total_amount" },
];

export default function ResultDisplay({ result }: ResultDisplayProps) {
  const ed = result.extracted_data;
  const ocr = result.ocr_metadata;
  const cfg = STATUS_CFG[result.status] || STATUS_CFG.failed;
  const StatusIcon = cfg.icon;

  return (
    <div className="space-y-5 animate-slide-up">
      {/* Status Bar */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold ${cfg.bg} ${cfg.color}`}>
          <StatusIcon className="w-3.5 h-3.5" /> {cfg.label}
        </span>
        <span className="text-sm text-txt-secondary">
          Confidence: <strong className="text-txt-primary">{(result.confidence_score * 100).toFixed(0)}%</strong>
        </span>
        <span className="text-sm text-txt-secondary">
          Fields: <strong className="text-txt-primary">{ed.fields_extracted}/{ed.fields_total}</strong>
        </span>
        <span className="text-sm">
          {result.validation_passed ? <span className="text-emerald-400">✅ Validated</span> : <span className="text-amber-400">⚠️ Issues</span>}
        </span>
        <span className="text-xs text-txt-muted px-2 py-0.5 bg-surface-secondary rounded border border-border">
          {ocr.engine_used}
        </span>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Metric label="Platform" value={(ed.platform || "N/A").toUpperCase()} accent="bg-blue-500" />
        <Metric label="Total Amount" value={ed.total_amount != null ? `₹${ed.total_amount}` : "N/A"} accent="bg-emerald-500" />
        <Metric label="Invoice No" value={ed.invoice_number || "N/A"} accent="bg-amber-500" />
        <Metric label="Processing" value={`${result.processing_time_seconds.toFixed(2)}s`} accent="bg-red-500" />
      </div>

      {/* Warnings */}
      {result.warnings.length > 0 ? (
        <div className="space-y-2">
          {result.warnings.map((w, i) => (
            <div key={i} className="flex items-start gap-2 px-4 py-3 bg-amber-500/10 border border-amber-500/20 rounded-lg text-sm text-amber-400">
              <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" /> {w}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2 px-4 py-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-sm text-emerald-400">
          <Shield className="w-4 h-4" /> All validations passed — data is consistent
        </div>
      )}

      {/* ── 22 MENTOR FIELDS ────────────────────────────────────── */}
      <div className="bg-surface-card border border-border rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-blue-400 uppercase tracking-wider">
            Fields
          </h3>
          <span className="text-xs px-2 py-0.5 bg-blue-500/10 border border-blue-500/20 rounded text-blue-400">
            {ed.fields_extracted}/22
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
          {MENTOR_FIELDS.map(({ key, label }) => {
            const val = (ed as any)[key];
            const filled = val !== null && val !== undefined && val !== "";
            return (
              <div key={key} className={`flex justify-between py-2 border-b border-border/30 text-sm ${filled ? "" : "opacity-40"}`}>
                <span className="text-txt-secondary font-mono text-xs">{label}</span>
                <span className={`text-right max-w-[55%] truncate ${filled ? "font-medium text-txt-primary" : "text-txt-muted italic"}`}>
                  {filled ? String(val) : "—"}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── BONUS FIELDS ────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div>
          <h3 className="text-xs font-semibold text-blue-400 uppercase tracking-wider mb-2">💰 Amounts</h3>
          <FieldRow label="Subtotal" value={ed.subtotal != null ? `₹${ed.subtotal}` : null} />
          <FieldRow label="CGST" value={ed.cgst_amount != null ? `₹${ed.cgst_amount}` : null} />
          <FieldRow label="SGST" value={ed.sgst_amount != null ? `₹${ed.sgst_amount}` : null} />
          <FieldRow label="IGST" value={ed.igst_amount != null ? `₹${ed.igst_amount}` : null} />
          <FieldRow label="Discount" value={ed.discount != null ? `₹${ed.discount}` : null} />
          <FieldRow label="Delivery" value={ed.delivery_charge != null ? `₹${ed.delivery_charge}` : null} />
          <FieldRow label="Packaging" value={ed.packaging_charge != null ? `₹${ed.packaging_charge}` : null} />
        </div>
        <div>
          <h3 className="text-xs font-semibold text-blue-400 uppercase tracking-wider mb-2">👤 Buyer & Payment</h3>
          <FieldRow label="Buyer Name" value={ed.buyer_name} />
          <FieldRow label="Phone" value={ed.buyer_phone} />
          <FieldRow label="Payment" value={ed.payment_method} />
          <div className="mt-4">
            <h3 className="text-xs font-semibold text-blue-400 uppercase tracking-wider mb-2">🔬 OCR Info</h3>
            <FieldRow label="Engine" value={ocr.engine_used} />
            <FieldRow label="OCR Confidence" value={`${(ocr.confidence * 100).toFixed(1)}%`} />
            <FieldRow label="Pages" value={ocr.pages_processed} />
            <FieldRow label="OCR Time" value={`${ocr.processing_time_ms.toFixed(0)}ms`} />
          </div>
        </div>
      </div>

      {/* Collapsibles */}
      <div className="space-y-3">
        <Collapsible title="Raw OCR Text" icon="📝">
          <pre className="p-4 text-xs font-mono text-txt-secondary whitespace-pre-wrap break-words max-h-80 overflow-y-auto bg-surface-input">
            {result.raw_text || "No raw text available"}
          </pre>
        </Collapsible>
        <Collapsible title="Full JSON Response" icon="📋">
          <pre className="p-4 text-xs font-mono text-txt-secondary whitespace-pre-wrap break-words max-h-96 overflow-y-auto bg-surface-input">
            {JSON.stringify(result, null, 2)}
          </pre>
        </Collapsible>
      </div>

      {/* ── DOWNLOAD BUTTONS ────────────────────────────────────── */}
      <div className="flex gap-3 flex-wrap">
        <button
          onClick={() => downloadExcel(ed)}
          className="flex items-center gap-2 px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-medium transition-colors"
        >
          <FileSpreadsheet className="w-4 h-4" /> Download Excel
        </button>
        <button
          onClick={() => { const csv = generateCSV([result]); downloadBlob(csv, `invoice_${ed.invoice_number || "export"}.csv`, "text/csv"); }}
          className="flex items-center gap-2 px-4 py-2.5 bg-surface-card border border-border rounded-lg text-sm font-medium hover:border-blue-500/50 transition-colors"
        >
          <Download className="w-4 h-4" /> CSV
        </button>
        <button
          onClick={() => { downloadBlob(JSON.stringify(result, null, 2), `invoice_${ed.invoice_number || "export"}.json`, "application/json"); }}
          className="flex items-center gap-2 px-4 py-2.5 bg-surface-card border border-border rounded-lg text-sm font-medium hover:border-blue-500/50 transition-colors"
        >
          <Download className="w-4 h-4" /> JSON
        </button>
      </div>
    </div>
  );
}