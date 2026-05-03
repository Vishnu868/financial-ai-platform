"use client";

import { useState, useEffect } from "react";
import {
  FileText, Package, MessageSquare, ClipboardList,
  Loader2, Wifi, WifiOff, Zap, ScanSearch, Building2,
} from "lucide-react";
import FileUpload from "@/components/FileUpload";
import ResultDisplay from "@/components/ResultDisplay";
import BatchPanel from "@/components/BatchPanel";
import ChatPanel from "@/components/ChatPanel";
import HistoryPanel from "@/components/HistoryPanel";
import YoloPanel from "@/components/YoloPanel";
import BankPanel from "@/components/BankPanel";
import { checkHealth, extractInvoice } from "@/lib/api";
import type { ExtractionResponse } from "@/types/api";

type Tab = "extract" | "batch" | "yolo" | "bank" | "chat" | "history";

const TABS: { id: Tab; label: string; icon: React.ReactNode; color: string }[] = [
  { id: "extract",  label: "Extract",  icon: <FileText      className="w-4 h-4" />, color: "bg-blue-600 shadow-blue-600/20" },
  { id: "batch",    label: "Batch",    icon: <Package       className="w-4 h-4" />, color: "bg-blue-600 shadow-blue-600/20" },
  { id: "yolo",     label: "YOLO",     icon: <ScanSearch    className="w-4 h-4" />, color: "bg-purple-600 shadow-purple-600/20" },
  { id: "bank",     label: "Bank",     icon: <Building2     className="w-4 h-4" />, color: "bg-emerald-600 shadow-emerald-600/20" },
  { id: "chat",     label: "Chat",     icon: <MessageSquare className="w-4 h-4" />, color: "bg-blue-600 shadow-blue-600/20" },
  { id: "history",  label: "History",  icon: <ClipboardList className="w-4 h-4" />, color: "bg-blue-600 shadow-blue-600/20" },
];

const PLATFORMS = [
  "Amazon", "Flipkart", "Swiggy", "Zomato", "Meesho",
  "Myntra", "BigBasket", "Blinkit", "JioMart",
];

export default function Home() {
  const [tab, setTab] = useState<Tab>("extract");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [result, setResult] = useState<ExtractionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<ExtractionResponse[]>([]);

  useEffect(() => {
    const check = async () => setApiOnline(await checkHealth());
    check();
    const interval = setInterval(check, 12000);
    return () => clearInterval(interval);
  }, []);

  const handleExtract = async () => {
    if (!file) return;
    setExtracting(true);
    setError(null);
    const { data, error: err } = await extractInvoice(file);
    if (err) setError(err);
    else if (data) { setResult(data); setHistory(p => [...p, data]); }
    setExtracting(false);
  };

  const activeTab = TABS.find(t => t.id === tab);

  return (
    <div className="min-h-screen bg-surface">
      {/* HEADER */}
      <header className="border-b border-border">
        <div className="max-w-[1400px] mx-auto px-6 py-5 flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white">
              <Zap className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-txt-primary">Financial Document AI</h1>
              <p className="text-[11px] text-txt-muted">Ensemble OCR · YOLO · GST Validation</p>
            </div>
          </div>
          <div className={`flex items-center gap-2 text-xs font-medium px-3.5 py-1.5 rounded-full border ${
            apiOnline === null ? "border-border text-txt-muted"
            : apiOnline ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
            : "border-red-500/30 bg-red-500/10 text-red-400"
          }`}>
            {apiOnline === null ? <Loader2 className="w-3 h-3 animate-spin" />
              : apiOnline ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {apiOnline === null ? "Checking..." : apiOnline ? "API Connected" : "API Offline"}
          </div>
        </div>
      </header>

      {/* MAIN */}
      <main className="max-w-[1400px] mx-auto px-6 py-6">
        {/* Tabs */}
        <div className="flex gap-1 p-1 bg-surface-secondary rounded-xl border border-border mb-7">
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex-1 flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                tab === t.id
                  ? `${t.color} text-white shadow-lg`
                  : "text-txt-secondary hover:text-txt-primary hover:bg-surface-card"
              }`}
            >
              {t.icon}
              <span className="hidden sm:inline">{t.label}</span>
            </button>
          ))}
        </div>

        {/* TAB: EXTRACT */}
        {tab === "extract" && (
          <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-6">
            <div className="space-y-5">
              <div className="bg-surface-card border border-border rounded-xl p-6">
                <h2 className="font-semibold mb-4 flex items-center gap-2 text-txt-primary">
                  <FileText className="w-5 h-5 text-blue-400" /> Upload Document
                </h2>
                <FileUpload
                  accept=".pdf,.jpg,.jpeg,.png,.webp"
                  label="Drop file here or click to browse"
                  hint="PDF, JPG, PNG, WebP"
                  selectedFile={file}
                  onFileSelect={setFile}
                  onClear={() => { setFile(null); setResult(null); setError(null); }}
                />
                <button
                  onClick={handleExtract}
                  disabled={!file || extracting || !apiOnline}
                  className="w-full mt-5 flex items-center justify-center gap-2 px-5 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-600/40 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
                >
                  {extracting ? <><Loader2 className="w-4 h-4 animate-spin" /> Extracting...</>
                    : <><Zap className="w-4 h-4" /> Extract Data</>}
                </button>
                {error && (
                  <div className="mt-4 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg text-sm text-red-400">{error}</div>
                )}
              </div>

            </div>
            <div>
              {result ? <ResultDisplay result={result} /> : (
                <div className="bg-surface-card border border-border rounded-xl p-16 text-center">
                  <p className="text-4xl mb-3">📄</p>
                  <p className="text-lg font-semibold text-txt-primary">Upload a document to get started</p>
                </div>
              )}
            </div>
          </div>
        )}

        {tab === "batch"   && <BatchPanel onResults={r => setHistory(p => [...p, ...r])} />}
        {tab === "yolo"    && <YoloPanel />}
        {tab === "bank"    && <BankPanel />}
        {tab === "chat"    && <ChatPanel result={result} />}
        {tab === "history" && <HistoryPanel history={history} onClear={() => setHistory([])} />}
      </main>
    </div>
  );
}