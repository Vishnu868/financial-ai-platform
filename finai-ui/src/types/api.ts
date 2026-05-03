// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Types matching FastAPI Pydantic schemas exactly
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export type DocumentPlatform =
  | "amazon" | "flipkart" | "meesho" | "myntra"
  | "swiggy" | "zomato" | "bigbasket" | "blinkit"
  | "jiomart" | "unknown";

export type DocumentType = "invoice" | "bank_statement" | "unknown";

export type ExtractionStatus = "success" | "partial" | "low_confidence" | "failed";

export interface TaxBreakdown {
  cgst_rate: number | null;
  cgst_amount: number | null;
  sgst_rate: number | null;
  sgst_amount: number | null;
  igst_rate: number | null;
  igst_amount: number | null;
  cess: number | null;
  total_tax: number | null;
  is_validated: boolean;
}

export interface SellerInfo {
  name: string | null;
  gstin: string | null;
  pan: string | null;
  address: string | null;
  state: string | null;
  state_code: string | null;
}

export interface BuyerInfo {
  name: string | null;
  address: string | null;
  phone: string | null;
  state: string | null;
}

export interface InvoiceItem {
  sr_no: number | null;
  description: string | null;
  hsn_code: string | null;
  quantity: number | null;
  unit_price: number | null;
  discount: number | null;
  tax_rate: number | null;
  total_price: number | null;
}

export interface InvoiceData {
  invoice_number: string | null;
  order_id: string | null;
  platform: DocumentPlatform;
  invoice_date: string | null;
  due_date: string | null;
  place_of_supply: string | null;
  seller: SellerInfo | null;
  buyer: BuyerInfo | null;
  items: InvoiceItem[] | null;
  subtotal: number | null;
  tax: TaxBreakdown | null;
  discount: number | null;
  delivery_charge: number | null;
  packaging_charge: number | null;
  total_amount: number | null;
  amount_in_words: string | null;
  currency: string;
  payment_method: string | null;
  validation_warnings: string[];
  fields_extracted: number;
  fields_total: number;
}

export interface OCRMetadata {
  engine_used: string;
  confidence: number;
  paddle_regions: number;
  easy_regions: number;
  merged_regions: number;
  processing_time_ms: number;
  fallback_used: boolean;
  image_preprocessed: boolean;
  pages_processed: number;
}

export interface ExtractionResponse {
  status: ExtractionStatus;
  message: string;
  document_type: DocumentType;
  platform: DocumentPlatform;
  confidence_score: number;
  extracted_data: InvoiceData;
  ocr_metadata: OCRMetadata;
  raw_text: string | null;
  processing_time_seconds: number;
  validation_passed: boolean;
  warnings: string[];
  extraction_id: string | null;
}

export interface BatchStatus {
  batch_id: string;
  total_files: number;
  completed: number;
  failed: number;
  status: "processing" | "completed" | "failed";
  results: ExtractionResponse[] | null;
  created_at: string | null;
}

export interface ClassificationResult {
  document_type: DocumentType;
  platform: DocumentPlatform;
  confidence: number;
  method: string;
  all_scores: Record<string, number>;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}
