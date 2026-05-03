"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2 } from "lucide-react";
import type { ExtractionResponse, ChatMessage } from "@/types/api";
import { chatWithOllama } from "@/lib/api";

interface ChatPanelProps {
  result: ExtractionResponse | null;
}

function buildContext(result: ExtractionResponse): string {
  const ed = result.extracted_data;
  const s = ed.seller || ({} as any);
  const t = ed.tax || ({} as any);
  const b = ed.buyer || ({} as any);

  return [
    `Platform: ${result.platform}`,
    `Invoice Number: ${ed.invoice_number}`,
    `Order ID: ${ed.order_id}`,
    `Date: ${ed.invoice_date}`,
    `Seller: ${s.name}`,
    `GSTIN: ${s.gstin}`,
    `Buyer: ${b.name}`,
    `Subtotal: ${ed.subtotal}`,
    `CGST: ${t.cgst_amount}`,
    `SGST: ${t.sgst_amount}`,
    `IGST: ${t.igst_amount}`,
    `Total Tax: ${t.total_tax}`,
    `Discount: ${ed.discount}`,
    `Delivery: ${ed.delivery_charge}`,
    `Total: ${ed.total_amount}`,
    `Payment: ${ed.payment_method}`,
    `Validated: ${result.validation_passed}`,
    `Warnings: ${result.warnings.join("; ")}`,
    `\nRaw OCR (first 1200 chars):\n${(result.raw_text || "").slice(0, 1200)}`,
  ].join("\n");
}

function keywordFallback(q: string, result: ExtractionResponse): string {
  const ed = result.extracted_data;
  const s = ed.seller || ({} as any);
  const t = ed.tax || ({} as any);
  const ql = q.toLowerCase();

  if (ql.includes("total") && (ql.includes("amount") || ql.includes("grand")))
    return `The total amount is ₹${ed.total_amount ?? "not found"}.`;
  if (ql.includes("invoice") && ql.includes("number"))
    return `The invoice number is ${ed.invoice_number || "not found"}.`;
  if (ql.includes("gstin")) return `The seller GSTIN is ${s.gstin || "not found"}.`;
  if (ql.includes("seller") || ql.includes("sold"))
    return `The seller is ${s.name || "not found"}.`;
  if (ql.includes("tax"))
    return `Tax breakdown: CGST ₹${t.cgst_amount ?? "N/A"}, SGST ₹${t.sgst_amount ?? "N/A"}, IGST ₹${t.igst_amount ?? "N/A"}, Total Tax ₹${t.total_tax ?? "N/A"}.`;
  if (ql.includes("date")) return `The invoice date is ${ed.invoice_date || "not found"}.`;
  if (ql.includes("payment")) return `Payment method: ${ed.payment_method || "not found"}.`;

  return `Here's what I know:\n• Platform: ${result.platform}\n• Total: ₹${ed.total_amount}\n• Date: ${ed.invoice_date}\n• Seller: ${s.name}\n\nAsk about specific fields like total, tax, GSTIN, seller, date, or payment.`;
}

export default function ChatPanel({ result }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "assistant", content: "Hi! Extract a document first, then ask me anything about it." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");

    setMessages((prev) => [...prev, { role: "user", content: q }]);

    if (!result) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Please extract a document first using the Extract tab." },
      ]);
      return;
    }

    setLoading(true);

    // Try Ollama first
    const context = buildContext(result);
    let answer = await chatWithOllama(q, context);

    // Fallback to keyword matching
    if (!answer) {
      answer = keywordFallback(q, result);
    }

    setMessages((prev) => [...prev, { role: "assistant", content: answer! }]);
    setLoading(false);
  };

  const quickQuestions = [
    "What is the total amount?",
    "Show tax breakdown",
    "Who is the seller?",
  ];

  return (
    <div className="flex flex-col h-[560px] border border-border rounded-xl overflow-hidden bg-surface-card">
      {/* Header */}
      <div className="px-5 py-3.5 border-b border-border bg-surface-secondary">
        <p className="text-sm font-semibold text-txt-primary flex items-center gap-2">
          <Bot className="w-4 h-4 text-blue-400" /> Document Chat
        </p>
        {result && (
          <p className="text-xs text-txt-muted mt-0.5">
            Chatting about: {result.extracted_data.invoice_number || "Uploaded Document"} ({result.platform?.toUpperCase()})
          </p>
        )}
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] px-4 py-2.5 rounded-xl text-sm leading-relaxed whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-blue-600 text-white rounded-br-sm"
                  : "bg-surface-secondary border border-border text-txt-primary rounded-bl-sm"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-surface-secondary border border-border rounded-xl px-4 py-2.5 rounded-bl-sm">
              <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
            </div>
          </div>
        )}
      </div>

      {/* Quick Questions */}
      {result && messages.length <= 2 && (
        <div className="px-5 py-2 flex gap-2 flex-wrap border-t border-border/50">
          {quickQuestions.map((q) => (
            <button
              key={q}
              onClick={() => {
                setInput(q);
                setTimeout(() => sendMessage(), 50);
              }}
              className="text-xs px-3 py-1.5 bg-surface-secondary border border-border rounded-full text-txt-secondary hover:text-txt-primary hover:border-blue-500/40 transition-colors"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="flex gap-2 p-4 border-t border-border bg-surface-secondary">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask about the document..."
          className="flex-1 px-4 py-2.5 bg-surface-input border border-border rounded-lg text-sm text-txt-primary placeholder-txt-muted outline-none focus:border-blue-500 transition-colors"
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="px-4 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-600/50 text-white rounded-lg transition-colors"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
