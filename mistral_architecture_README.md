# Mistral Architecture — Complete Study Guide

> Study reference for Financial Document AI Platform internship.  
> Every concept explained from first principles. Enough depth to answer any question your sir may ask.

---

## Table of Contents

1. [What is Mistral?](#1-what-is-mistral)
2. [High-Level Architecture Overview](#2-high-level-architecture-overview)
3. [Tokenization](#3-tokenization)
4. [Embedding Layer](#4-embedding-layer)
5. [Transformer Decoder Blocks](#5-transformer-decoder-blocks)
6. [Grouped Query Attention (GQA)](#6-grouped-query-attention-gqa)
7. [Sliding Window Attention (SWA)](#7-sliding-window-attention-swa)
8. [Rolling Buffer KV Cache](#8-rolling-buffer-kv-cache)
9. [RMSNorm — Pre-Normalization](#9-rmsnorm--pre-normalization)
10. [SwiGLU Activation in FFN](#10-swiglu-activation-in-ffn)
11. [Rotary Positional Embeddings (RoPE)](#11-rotary-positional-embeddings-rope)
12. [No Bias Terms](#12-no-bias-terms)
13. [Mistral 7B vs Llama 2 7B — Key Differences](#13-mistral-7b-vs-llama-2-7b--key-differences)
14. [Mixtral — Mixture of Experts Extension](#14-mixtral--mixture-of-experts-extension)
15. [How Mistral Generates Text (Inference Flow)](#15-how-mistral-generates-text-inference-flow)
16. [Why Mistral is Used in This Project (Ollama RAG)](#16-why-mistral-is-used-in-this-project-ollama-rag)
17. [Important Numbers to Remember](#17-important-numbers-to-remember)
18. [Common Interview Questions & Answers](#18-common-interview-questions--answers)

---

## 1. What is Mistral?

**Mistral 7B** is a large language model (LLM) released by Mistral AI in September 2023. It has **7.3 billion parameters** and outperforms Meta's Llama 2 13B (which is nearly twice its size) on most benchmarks.

Key facts:
- Released: September 2023
- Parameters: 7.3 billion
- Context length: 8,192 tokens (with SWA can extrapolate longer)
- License: Apache 2.0 (fully open — can use commercially for free)
- Architecture base: Transformer decoder-only (same family as GPT, Llama)
- Main innovations: **Grouped Query Attention**, **Sliding Window Attention**, **Rolling Buffer Cache**

Think of it as a smarter, faster, smaller model — it proves that architectural innovations matter more than just scaling up parameters.

---

## 2. High-Level Architecture Overview

```
Input Text
    │
    ▼
[Tokenizer — BPE, 32,000 vocab]
    │
    ▼
[Token Embedding Layer]   ← maps token IDs to 4096-dim vectors
    │
    ▼
┌─────────────────────────────────────┐
│  Transformer Decoder Block × 32     │ ← repeated 32 times
│                                     │
│  ┌─────────────────────────────┐    │
│  │  RMSNorm (pre-norm)         │    │
│  └──────────┬──────────────────┘    │
│             ▼                       │
│  ┌─────────────────────────────┐    │
│  │  Grouped Query Attention    │    │
│  │  (with Sliding Window)      │    │
│  │  + RoPE positional encoding │    │
│  └──────────┬──────────────────┘    │
│             │ + residual            │
│             ▼                       │
│  ┌─────────────────────────────┐    │
│  │  RMSNorm (pre-norm)         │    │
│  └──────────┬──────────────────┘    │
│             ▼                       │
│  ┌─────────────────────────────┐    │
│  │  Feed-Forward Network       │    │
│  │  (SwiGLU activation)        │    │
│  └──────────┬──────────────────┘    │
│             │ + residual            │
└─────────────┼───────────────────────┘
              ▼
[RMSNorm — final]
    │
    ▼
[Linear projection → 32,000 logits]
    │
    ▼
[Softmax → probability distribution]
    │
    ▼
Next token predicted
```

The model is **decoder-only** — it generates one token at a time, left to right. Each new token attends to all previous tokens in the sequence.

---

## 3. Tokenization

Mistral uses **Byte-Pair Encoding (BPE)** with a vocabulary size of **32,000 tokens**.

### What is BPE?
BPE starts from individual characters and merges the most frequent character pairs repeatedly until it reaches the vocabulary size.

Example:
```
"invoice" → ["in", "vo", "ice"] (BPE might split it into subwords)
"GSTIN"   → ["G", "ST", "IN"]  (rare word → character-level fallback)
```

### Why BPE?
- Handles unknown words by breaking into subwords
- More efficient than character-level (fewer tokens per sentence)
- Less vocabulary needed than word-level

### SentencePiece
Mistral uses Google's SentencePiece library for tokenization, which handles whitespace as part of tokens (no need for special whitespace markers).

---

## 4. Embedding Layer

After tokenization, each token ID is looked up in an **embedding matrix** of shape `[32000, 4096]`.

- `32000` = vocabulary size
- `4096` = hidden dimension (d_model)

Each token becomes a 4096-dimensional vector. This is a learned representation — similar words/tokens end up with similar vectors in this high-dimensional space.

The embedding weights are also **tied** with the output projection layer (same matrix used at input and output) — this saves parameters and improves training.

---

## 5. Transformer Decoder Blocks

Mistral has **32 decoder layers**. Each layer has two main sub-components:

### Sub-component 1: Self-Attention
- Lets every token "look at" other tokens and decide what's relevant
- Uses Grouped Query Attention (see section 6)
- Uses Sliding Window (see section 7)
- Uses RoPE for positions (see section 11)

### Sub-component 2: Feed-Forward Network (FFN)
- Applies a non-linear transformation to each token independently
- Uses SwiGLU activation (see section 10)
- Hidden dimension: **14336** (much larger than d_model=4096)

### Residual Connections
After each sub-component, the input is **added back** to the output:
```
output = sub_component(RMSNorm(input)) + input
```

This is called a **residual connection** (or skip connection). It prevents vanishing gradients during training and allows the model to easily learn identity functions (ignore the layer if needed).

### Pre-Normalization vs Post-Normalization
Original Transformers applied LayerNorm **after** the attention/FFN. Mistral (like Llama) applies **RMSNorm before** — this is called Pre-Norm and leads to more stable training.

---

## 6. Grouped Query Attention (GQA)

This is one of Mistral's most important innovations for **speed and memory efficiency**.

### Standard Multi-Head Attention (MHA) — the old way
In a regular Transformer:
- 32 Query heads (Q)
- 32 Key heads (K)
- 32 Value heads (V)

During inference, the KV cache stores all previous Keys and Values for every head. For 32 heads × large sequence → huge memory usage.

### Multi-Query Attention (MQA) — too aggressive
- 32 Query heads
- 1 Key head (all queries share same K)
- 1 Value head (all queries share same V)

This saves memory but hurts quality — one K/V for 32 queries is too much sharing.

### Grouped Query Attention (GQA) — Mistral's approach
- 32 Query heads
- **8 Key heads** (groups of 4 queries share one K head)
- **8 Value heads** (groups of 4 queries share one V head)

```
Q heads:  [Q1  Q2  Q3  Q4] [Q5  Q6  Q7  Q8] ... (32 total)
                 │                │
K heads:       [K1]            [K2]            ... (8 total)
V heads:       [V1]            [V2]            ... (8 total)
```

**Result:**
- Quality close to full MHA (queries still have diversity)
- Memory 4× smaller than MHA (8 KV heads vs 32)
- Inference speed significantly faster

### Mistral's exact numbers:
- `n_heads` = 32 (query heads)
- `n_kv_heads` = 8 (key-value heads)
- `head_dim` = 128 (4096 / 32)

---

## 7. Sliding Window Attention (SWA)

### The Problem: Quadratic Attention Cost
Standard attention: every token attends to ALL previous tokens.  
Cost = O(sequence_length²) — extremely slow and memory-hungry for long documents.

### Sliding Window Attention Solution
Each token only attends to the **W most recent tokens** (W = window size).

Mistral uses **W = 4096 tokens** per layer.

```
Standard attention (token at position 8 sees all):
  pos: 1  2  3  4  5  6  7  [8]
         ↑  ↑  ↑  ↑  ↑  ↑  ↑

Sliding window (W=4, token at position 8 sees only last 4):
  pos: 1  2  3  4  5  6  7  [8]
                  ↑  ↑  ↑  ↑
```

### But Wait — Doesn't This Lose Long-Range Context?

No! Because there are **32 layers** and each layer attends W tokens. Information **propagates** through layers:

- Layer 1: token 8 sees tokens 5–8
- Layer 2: token 8 (updated with 5–8 info) sees tokens 5–8 again, but token 7 already has info from tokens 4–7
- After all 32 layers: token 8 has effectively "seen" up to 32 × 4096 = **131,072 tokens** ago

This is called the **receptive field** growing through layers — similar to how a convolutional neural network sees a small patch at layer 1 but a large patch at layer 5.

### Attention Sink
There's a special modification: the **first 4 tokens** (position 0–3) are always included in every window, regardless of distance. These "sink" tokens are important because early tokens (like system prompts) encode crucial context that all later tokens need.

---

## 8. Rolling Buffer KV Cache

This is a memory optimization for inference time.

### The Problem
During text generation, you need to cache all previous K and V values to avoid recomputing them. For a 8K context, this is a huge buffer.

### Rolling Buffer Solution
Instead of a growing buffer, Mistral uses a **fixed-size circular buffer** of size W (= 4096).

```
Buffer position = sequence_position % W

Position 0    → slot 0
Position 1    → slot 1
...
Position 4095 → slot 4095
Position 4096 → slot 0  (overwrites oldest!)
Position 4097 → slot 1  (overwrites oldest!)
```

This keeps memory usage **constant** regardless of sequence length. Old tokens fall out of the window anyway (due to SWA), so overwriting them is fine.

**Memory savings:** Instead of O(sequence_length) cache, it's always O(W) = constant.

---

## 9. RMSNorm — Pre-Normalization

Mistral uses **Root Mean Square Layer Normalization** instead of standard LayerNorm.

### Standard LayerNorm formula:
```
LayerNorm(x) = γ × (x - mean(x)) / √(var(x) + ε) + β
```

### RMSNorm formula:
```
RMSNorm(x) = γ × x / √(mean(x²) + ε)
```

Key differences:
1. **No mean subtraction** — removes the "re-centering" step
2. **No β bias** — removes the learnable shift
3. Only has **γ** (scale parameter)

### Why RMSNorm?
- **Faster:** No need to compute mean separately
- **Simpler:** Fewer parameters (no β)
- **Works just as well** as LayerNorm in practice for LLMs
- About **10–15% faster** than LayerNorm in benchmarks

Applied **before** attention and FFN (Pre-Norm) for training stability.

---

## 10. SwiGLU Activation in FFN

The Feed-Forward Network in each Transformer block uses **SwiGLU** activation.

### Traditional FFN (GPT-style):
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```
(ReLU activation)

### Mistral's FFN with SwiGLU:
```python
def FFN_SwiGLU(x):
    gate = x @ W_gate    # shape: [seq, 14336]
    up   = x @ W_up      # shape: [seq, 14336]
    down_input = SiLU(gate) * up   # element-wise
    return down_input @ W_down     # shape: [seq, 4096]
```

### SiLU (Sigmoid Linear Unit):
```
SiLU(x) = x × sigmoid(x) = x / (1 + e^(-x))
```
SwiGLU = SiLU(gate) × up → a **gating mechanism** where the gate controls how much information from `up` passes through.

### Why SwiGLU?
- Better gradient flow than ReLU
- The gating mechanism gives the FFN more expressive power
- Empirically improves model quality significantly
- Used in PaLM, LLaMA, and now Mistral

### Dimensions:
- Input/output: 4096 (d_model)
- Hidden: **14336** (= 4096 × ~3.5)
- Three weight matrices: W_gate, W_up, W_down

---

## 11. Rotary Positional Embeddings (RoPE)

Transformers have no built-in sense of order — without position encoding, "cat sat on mat" and "mat on sat cat" look identical. Mistral uses **RoPE** to encode positions.

### The Core Idea
Instead of adding a positional vector to embeddings (as in original Transformers), RoPE **rotates** the Query and Key vectors in 2D planes based on their position.

For a vector at position `m`, rotate pairs of dimensions by angle `m × θ_i`:
```
[q₁, q₂] → [q₁cos(mθ) - q₂sin(mθ), q₁sin(mθ) + q₂cos(mθ)]
```

This is done for all pairs of dimensions with different rotation frequencies `θ_i`.

### Why Rotation Works for Attention
The dot product Q·K (which is what attention computes) between position m and position n naturally encodes the **relative distance** (m - n):

```
RoPE(Q_m) · RoPE(K_n) = f(Q, K, m - n)
```

The model learns relative positions automatically — position 1000 relative to position 995 gives the same signal as position 5 relative to position 0, as long as the gap is 5.

### Why RoPE over Original Position Embeddings?
1. **Relative positions:** Model understands "5 tokens ago" not just "position 47"
2. **Length extrapolation:** Can generalize to longer sequences than seen during training
3. **No extra parameters:** Purely deterministic rotation, nothing to learn
4. **Works with SWA:** Naturally fits windowed attention since attention is position-relative

### Mistral's RoPE settings:
- `rope_theta` = 10000 (base frequency)
- Applied only to Q and K, not V

---

## 12. No Bias Terms

Mistral removes bias terms from all linear layers (attention projections, FFN weights).

Standard linear: `y = xW + b`  
Mistral's linear: `y = xW`

### Why?
- Pre-normalization (RMSNorm) already handles centering
- Bias terms add parameters without proportional benefit
- Slightly faster computation
- Models of this scale don't need bias to represent complex functions

---

## 13. Mistral 7B vs Llama 2 7B — Key Differences

| Feature | Mistral 7B | Llama 2 7B |
|---|---|---|
| Parameters | 7.3B | 6.7B |
| Context length | 8,192 | 4,096 |
| Attention type | Grouped Query (8 KV heads) | Multi-Head (32 KV heads) |
| Sliding Window | Yes (W=4096) | No |
| KV Cache | Rolling Buffer | Standard growing cache |
| FFN activation | SwiGLU | SwiGLU |
| Normalization | RMSNorm | RMSNorm |
| Position encoding | RoPE | RoPE |
| Memory at inference | Much lower | Higher |
| Speed | Faster | Slower |
| Quality (benchmarks) | Beats Llama 2 13B | Llama 2 13B level |

Mistral 7B is smarter than its size suggests because of architectural efficiency, not just more parameters.

---

## 14. Mixtral — Mixture of Experts Extension

Mistral AI also released **Mixtral 8×7B** — an extension using Mixture of Experts (MoE).

### Core Idea
Instead of one FFN in each Transformer block, there are **8 FFN "experts"**. For each token, a **router** picks the top **2 experts** to process that token.

```
Token → Router → selects Expert 3 and Expert 7
Output = w₃ × Expert3(token) + w₇ × Expert7(token)
```

### Why MoE?
- Total parameters: 8 × 7B = ~46B (much larger model)
- Active parameters per token: only 2 experts × 7B = ~13B (only 2 are used)
- You get a 46B-quality model at the inference cost of a 13B model
- Different experts specialize in different types of knowledge

### Mistral's MoE specs:
- 8 experts per layer
- 2 experts active per token
- 32 layers
- Total: ~46.7B parameters
- Active per token: ~13B parameters

---

## 15. How Mistral Generates Text (Inference Flow)

This is the complete process from prompt to output:

```
Step 1: Tokenize the input prompt
"What is the total amount?" → [1045, 338, 278, 3001, 5253, 29973]

Step 2: Embed tokens → 6 vectors of shape [4096]

Step 3: Pass through 32 decoder layers
  For each layer:
    a. RMSNorm the input
    b. Compute Q, K, V projections
    c. Apply RoPE to Q and K
    d. Look up rolling KV cache for previous tokens
    e. Add current K,V to cache
    f. Compute attention with sliding window (attend to last 4096 tokens)
    g. Project attention output
    h. Add residual
    i. RMSNorm
    j. SwiGLU FFN
    k. Add residual

Step 4: Apply final RMSNorm

Step 5: Project to vocabulary size [32000]
  Linear layer: [4096] → [32000] (logits)

Step 6: Apply temperature and sample
  - Temperature < 1.0 → sharper distribution (more focused)
  - Temperature > 1.0 → flatter distribution (more creative)
  - Softmax → probabilities
  - Sample or take argmax → next token ID

Step 7: Decode token ID back to text

Step 8: Append to sequence, repeat from step 3 for next token
  (previous tokens' K,V values are cached — not recomputed)
```

This continues until an `<EOS>` (end-of-sequence) token is generated or max_tokens is reached.

---

## 16. Why Mistral is Used in This Project (Ollama RAG)

In the Financial Document AI platform (`rag_engine.py`):

```python
OLLAMA_MODEL = "mistral"
llm = Ollama(model=OLLAMA_MODEL, temperature=0.1)
answer = llm.invoke(prompt)
```

### Why Mistral specifically?
1. **Runs locally via Ollama** — no API key, no internet, no cost
2. **Small enough** for CPU inference (quantized to 4-bit = ~4GB RAM)
3. **Strong instruction following** — understands "Answer only from document"
4. **Good at factual extraction** — doesn't hallucinate much at temperature=0.1
5. **Apache 2.0 license** — can use commercially

### Quantization in Ollama
Ollama serves Mistral in **GGUF format** with 4-bit quantization (Q4_K_M):
- Original: 7.3B params × 16-bit float = ~14.6GB
- Quantized: ~4.1GB — runs on any 8GB RAM laptop

Quantization reduces precision (16-bit → 4-bit) with minimal quality loss for inference.

### The RAG flow with Mistral:
```
Invoice PDF
    → OCR (RapidOCR / EasyOCR)
    → Text chunks (400 chars)
    → Embeddings (all-MiniLM-L6-v2, local)
    → ChromaDB vector store
    → User asks question
    → Find top-3 relevant chunks
    → Build prompt with chunks + structured fields
    → Mistral (via Ollama) generates answer
    → Answer shown in Streamlit UI
```

---

## 17. Important Numbers to Remember

| Parameter | Value |
|---|---|
| Parameters | 7.3 billion |
| Layers (depth) | 32 |
| Hidden dimension (d_model) | 4096 |
| FFN hidden dimension | 14336 |
| Attention heads (Q) | 32 |
| KV heads (GQA) | 8 |
| Head dimension | 128 |
| Vocabulary size | 32,000 |
| Context length | 8,192 |
| Sliding window size | 4,096 |
| RoPE theta | 10,000 |
| Release date | September 2023 |
| License | Apache 2.0 |

---

## 18. Common Interview Questions & Answers

**Q: What is the main difference between Mistral and a standard Transformer?**  
A: Mistral introduces three key innovations over a standard Transformer decoder: Grouped Query Attention (reduces KV heads from 32 to 8, saving memory and speeding up inference), Sliding Window Attention (each token only attends to the last 4096 tokens rather than all tokens, making long context efficient), and a Rolling Buffer KV Cache (fixed-size circular cache instead of growing memory). It also uses RMSNorm instead of LayerNorm and SwiGLU activation instead of ReLU.

**Q: Explain Sliding Window Attention in simple terms.**  
A: Normally, every word in a sentence looks at every other word — if the document is 10,000 words, that's 10,000 × 10,000 = 100 million operations, which is slow and expensive. SWA restricts each word to only look at the nearest 4,096 words. But because there are 32 layers stacked, information from distant words still reaches every position indirectly — like passing a message through a chain of people. The effective reach after 32 layers is 32 × 4,096 ≈ 131,000 tokens.

**Q: What is Grouped Query Attention and why does it matter?**  
A: In standard multi-head attention, if you have 32 query heads, you also need 32 separate key and value heads — this means caching 32 × 2 = 64 matrices during inference, which uses a lot of GPU memory. GQA reduces the KV heads to just 8 — groups of 4 queries share one K and one V head. This reduces the KV cache memory by 4× with minimal quality loss, allowing faster inference and support for longer sequences.

**Q: Why does Mistral use RMSNorm instead of LayerNorm?**  
A: RMSNorm is simpler (removes mean subtraction and the β parameter) and faster (about 10–15% speedup) while achieving the same training stability. At billion-parameter scale, every small efficiency gain compounds significantly.

**Q: What is SwiGLU and why is it better than ReLU?**  
A: SwiGLU = SiLU(gate) × up. It uses a gating mechanism where one linear projection controls how much of another projection passes through. Compared to ReLU, SwiGLU provides smoother gradients (no hard zero cutoff), better expressivity through the gate, and empirically higher model quality across most benchmarks.

**Q: What is the difference between Mistral 7B and Mixtral 8×7B?**  
A: Mistral 7B is a standard dense model with 7.3B parameters — all parameters are used for every token. Mixtral 8×7B is a Mixture of Experts model: it has 8 parallel FFN experts in each layer, but only 2 are activated per token. Total parameters are ~46B, but only ~13B are active — giving the quality of a 46B model at the inference speed of a 13B model.

**Q: What is RoPE and how is it different from original positional embeddings?**  
A: Original positional embeddings (sinusoidal or learned) add a position vector to the token embedding — the model must then figure out relative distances from absolute positions. RoPE instead rotates the Q and K vectors by an angle proportional to their position. The key insight is that the attention dot product Q·K automatically becomes a function of the relative distance (m-n), not absolute positions. This makes RoPE better at length generalization and more naturally compatible with sliding window attention.

**Q: Why does Mistral outperform Llama 2 13B despite having fewer parameters?**  
A: Architecture efficiency. Mistral 7B uses GQA (faster attention), SWA (handles longer contexts efficiently), SwiGLU FFN (more expressive), RMSNorm (faster normalization), and was trained on higher-quality data with better training recipes. Having fewer parameters also means less overfitting and more generalization per parameter.

**Q: In your project, where exactly is Mistral used?**  
A: In the RAG chatbot (`rag_engine.py`). After OCR extracts text from the invoice PDF, the text is chunked into 400-character pieces and embedded using sentence-transformers (all-MiniLM-L6-v2). When the user asks a question, the top-3 most relevant chunks are retrieved from ChromaDB and passed to Mistral (via Ollama) as context. Mistral then generates an answer grounded in those document chunks. Temperature is set to 0.1 to keep answers factual and reduce hallucination.

---

*End of Mistral Architecture Study Guide*
