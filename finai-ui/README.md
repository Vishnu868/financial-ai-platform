# Financial Document AI — Next.js Frontend

Production-grade TypeScript + Next.js 14 UI for the Financial Document AI Platform.

## Features

| Tab | Features |
|-----|----------|
| **Extract** | Drag-and-drop upload, image preview, real-time extraction with metrics, validation warnings, OCR metadata, raw text toggle, CSV/JSON download |
| **Batch** | ZIP upload, progress bar with polling, results table, bulk CSV export |
| **Chat** | Ask questions about extracted data, Ollama (Mistral) integration with keyword fallback |
| **History** | Session extraction log as table, bulk CSV download, clear |

## Tech Stack

- **Next.js 14** (App Router)
- **TypeScript** (strict mode)
- **Tailwind CSS** (dark theme)
- **Lucide React** (icons)
- **Zero external UI libraries** — no shadcn, no Material UI, just Tailwind

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Set API URL (optional — defaults to localhost:8000)
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# 3. Run development server
npm run dev

# 4. Open http://localhost:3000
```

Make sure the FastAPI backend is running:
```bash
cd ../financial-ai-platform
uvicorn app.main:app --reload
```

## Production Build

```bash
npm run build
npm start
```

## Deployment

### Vercel (recommended)
```bash
npx vercel
```
Set `NEXT_PUBLIC_API_URL` in Vercel environment variables.

### Docker
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
CMD ["node", "server.js"]
```

### Static Export
```bash
# Add to next.config.mjs: output: 'export'
npm run build
# Deploy the 'out' folder to any static host
```

## Project Structure

```
finai-ui/
├── src/
│   ├── app/
│   │   ├── globals.css         # Tailwind + custom CSS
│   │   ├── layout.tsx          # Root layout with fonts
│   │   └── page.tsx            # Main page (4 tabs)
│   ├── components/
│   │   ├── FileUpload.tsx      # Drag-and-drop file upload
│   │   ├── ResultDisplay.tsx   # Extraction results (metrics, fields, warnings)
│   │   ├── BatchPanel.tsx      # Batch processing with progress
│   │   ├── ChatPanel.tsx       # Chat with Ollama + keyword fallback
│   │   └── HistoryPanel.tsx    # Extraction history table
│   ├── lib/
│   │   └── api.ts              # API client (all FastAPI endpoints + CSV generation)
│   └── types/
│       └── api.ts              # TypeScript types matching Pydantic schemas
├── .env.local                  # NEXT_PUBLIC_API_URL
├── tailwind.config.ts
├── tsconfig.json
├── next.config.mjs
└── package.json
```

## API Connection

The UI connects to FastAPI at `NEXT_PUBLIC_API_URL` (default: `http://localhost:8000`).

Endpoints used:
- `GET /health` — health check (polled every 12s)
- `POST /api/v2/invoice/extract` — single invoice extraction
- `POST /api/v2/batch/process` — start batch
- `GET /api/v2/batch/status/{id}` — poll batch
- `POST http://localhost:11434/api/generate` — Ollama chat (direct)
