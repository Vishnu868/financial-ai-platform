"use client";

import { Download, Trash2 } from "lucide-react";
import type { ExtractionResponse } from "@/types/api";
import { generateCSV, downloadBlob } from "@/lib/api";

interface HistoryPanelProps {
  history: ExtractionResponse[];
  onClear: () => void;
}

export default function HistoryPanel({ history, onClear }: HistoryPanelProps) {
  if (history.length === 0) {
    return (
      <div className="bg-surface-card border border-border rounded-xl p-12 text-center">
        <p className="text-3xl mb-3">📋</p>
        <p className="font-semibold text-txt-primary">No extractions yet</p>
        <p className="text-sm text-txt-muted mt-1">Upload a document to get started</p>
      </div>
    );
  }

  return (
    <div className="bg-surface-card border border-border rounded-xl overflow-hidden">
      <div className="px-6 py-4 border-b border-border flex justify-between items-center">
        <h3 className="font-semibold">Extraction History ({history.length})</h3>
        <div className="flex gap-2">
          <button
            onClick={() => {
              const csv = generateCSV(history);
              downloadBlob(csv, `all_extractions_${Date.now()}.csv`, "text/csv");
            }}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 bg-surface-secondary border border-border rounded-lg hover:border-blue-500/40 transition-colors"
          >
            <Download className="w-3.5 h-3.5" /> Download All CSV
          </button>
          <button
            onClick={onClear}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 bg-surface-secondary border border-red-500/30 rounded-lg text-red-400 hover:bg-red-500/10 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" /> Clear
          </button>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-surface-secondary text-left">
              {["#", "Platform", "Invoice", "Total", "Date", "Confidence", "Validated", "Status"].map((h) => (
                <th key={h} className="px-4 py-3 text-[11px] font-semibold text-txt-muted uppercase tracking-wider">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {history.map((r, i) => {
              const ed = r.extracted_data;
              return (
                <tr key={i} className="border-t border-border/40 hover:bg-surface-hover transition-colors">
                  <td className="px-4 py-3 text-txt-muted">{i + 1}</td>
                  <td className="px-4 py-3 font-medium">{r.platform?.toUpperCase()}</td>
                  <td className="px-4 py-3">{ed.invoice_number || "N/A"}</td>
                  <td className="px-4 py-3 text-emerald-400 font-medium">
                    {ed.total_amount != null ? `₹${ed.total_amount}` : "N/A"}
                  </td>
                  <td className="px-4 py-3 text-txt-secondary">{ed.invoice_date || "—"}</td>
                  <td className="px-4 py-3">{(r.confidence_score * 100).toFixed(0)}%</td>
                  <td className="px-4 py-3">{r.validation_passed ? "✅" : "⚠️"}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${
                        r.status === "success"
                          ? "bg-emerald-500/10 text-emerald-400"
                          : r.status === "partial"
                          ? "bg-amber-500/10 text-amber-400"
                          : "bg-red-500/10 text-red-400"
                      }`}
                    >
                      {r.status}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
