# LLaMA Architecture — Complete Study Guide

> Study reference for Financial Document AI Platform internship.  
> Covers LLaMA 1, LLaMA 2, and LLaMA 3. Every concept from first principles.

---

## Table of Contents

1. [What is LLaMA?](#1-what-is-llama)
2. [The LLaMA Family — Versions Overview](#2-the-llama-family--versions-overview)
3. [High-Level Architecture Diagram](#3-high-level-architecture-diagram)
4. [Tokenization in LLaMA](#4-tokenization-in-llama)
5. [Input Embeddings](#5-input-embeddings)
6. [The Transformer Decoder Block](#6-the-transformer-decoder-block)
7. [Multi-Head Self-Attention (LLaMA 1 & 2)](#7-multi-head-self-attention-llama-1--2)
8. [Grouped Query Attention (LLaMA 2 70B & LLaMA 3)](#8-grouped-query-attention-llama-2-70b--llama-3)
9. [Rotary Positional Embeddings (RoPE)](#9-rotary-positional-embeddings-rope)
10. [RMSNorm — Pre-Normalization](#10-rmsnorm--pre-normalization)
11. [SwiGLU Feed-Forward Network](#11-swiglu-feed-forward-network)
12. [KV Cache in LLaMA](#12-kv-cache-in-llama)
13. [Context Length Evolution](#13-context-length-evolution)
14. [LLaMA 1 vs LLaMA 2 vs LLaMA 3 — Full Comparison](#14-llama-1-vs-llama-2-vs-llama-3--full-comparison)
15. [LLaMA vs Mistral — Architectural Differences](#15-llama-vs-mistral--architectural-differences)
16. [Instruction Tuning — LLaMA-Chat / Instruct Models](#16-instruction-tuning--llama-chat--instruct-models)
17. [RLHF in LLaMA 2](#17-rlhf-in-llama-2)
18. [LLaMA 3 Specific Changes](#18-llama-3-specific-changes)
19. [How LLaMA Generates Text — Full Inference Flow](#19-how-llama-generates-text--full-inference-flow)
20. [Important Numbers to Remember](#20-important-numbers-to-remember)
21. [Common Interview Questions & Answers](#21-common-interview-questions--answers)

---

## 1. What is LLaMA?

**LLaMA** stands for **Large Language Model Meta AI**. It is a family of open-source large language models released by Meta (Facebook's parent company).

Timeline:
- **LLaMA 1** — February 2023. Sizes: 7B, 13B, 33B, 65B. Research-only license.
- **LLaMA 2** — July 2023. Sizes: 7B, 13B, 34B, 70B. Open commercial use (with restrictions).
- **LLaMA 3** — April 2024. Sizes: 8B, 70B (and later 405B). Truly open, state-of-the-art.
- **LLaMA 3.1/3.2** — Later 2024. Multimodal (vision) and 128K context support.

Why LLaMA matters:
- First high-quality open-weight model that researchers could actually use
- Sparked the entire open-source LLM ecosystem (Alpaca, Vicuna, Mistral, Falcon all built on its ideas)
- Proved you don't need GPT-4 scale to do useful things

---

## 2. The LLaMA Family — Versions Overview

### LLaMA 1 (Feb 2023)
- Released as research weights (leaked, then officially released)
- Sizes: 7B, 13B, 33B, 65B
- Context: 2,048 tokens
- Training data: ~1 trillion tokens from publicly available data
- Key contributions: Showed that smaller models trained on more data beat larger models trained on less data (Chinchilla scaling laws)

### LLaMA 2 (July 2023)
- Commercial license (free for most companies <700M monthly users)
- Sizes: 7B, 13B, 34B, 70B
- Context: 4,096 tokens
- Training data: ~2 trillion tokens
- Key additions: GQA for 70B model, RLHF for Chat models, safety improvements
- Comes in base (pretrained) and Chat (instruction-tuned) variants

### LLaMA 3 (April 2024)
- Sizes: 8B, 70B (405B later)
- Context: 8,192 tokens (base), 128K with RoPE scaling tricks
- Training data: 15 trillion tokens (massive increase)
- Key additions: New tokenizer (128K vocabulary), GQA for all sizes, improved instruction tuning
- 8B model beats LLaMA 2 70B across most benchmarks

### LLaMA 3.1 / 3.2 / 3.3
- 3.1: 405B model, 128K context, multilingual
- 3.2: Multimodal (vision) models 11B and 90B
- 3.3: Improved 70B with instruction tuning

---

## 3. High-Level Architecture Diagram

LLaMA is a **decoder-only Transformer** — same family as GPT. It generates tokens one at a time, left to right.

```
Input Text
    │
    ▼
[Tokenizer — SentencePiece BPE]
│  LLaMA 1&2: 32,000 vocab
│  LLaMA 3:   128,000 vocab
    │
    ▼
[Token Embedding Table]
│  Shape: [vocab_size, hidden_dim]
│  LLaMA 2 7B: [32000, 4096]
│  LLaMA 3 8B: [128000, 4096]
    │
    ▼
┌──────────────────────────────────────────────────┐
│   Decoder Layer × N  (N=32 for 7B/8B models)    │
│                                                  │
│   ┌────────────────────────────────────┐         │
│   │  RMSNorm (pre-attention)           │         │
│   └──────────────┬─────────────────────┘         │
│                  ▼                               │
│   ┌────────────────────────────────────┐         │
│   │  Self-Attention                    │         │
│   │  (MHA for small models,            │         │
│   │   GQA for large models)            │         │
│   │  + RoPE positional encoding        │         │
│   │  + KV Cache during inference       │         │
│   └──────────────┬─────────────────────┘         │
│                  │ + residual connection          │
│                  ▼                               │
│   ┌────────────────────────────────────┐         │
│   │  RMSNorm (pre-FFN)                 │         │
│   └──────────────┬─────────────────────┘         │
│                  ▼                               │
│   ┌────────────────────────────────────┐         │
│   │  Feed-Forward Network (SwiGLU)     │         │
│   │  3 linear layers, no bias          │         │
│   └──────────────┬─────────────────────┘         │
│                  │ + residual connection          │
└──────────────────┼───────────────────────────────┘
                   ▼
         [Final RMSNorm]
                   │
                   ▼
         [Output Linear Layer]
         [vocab_size logits]
                   │
                   ▼
         [Softmax + Sample → Next Token]
```

**Key insight:** There is NO encoder. LLaMA is decoder-only — it reads and generates all in one pass, using causal (masked) attention so each token can only see previous tokens, not future ones.

---

## 4. Tokenization in LLaMA

### LLaMA 1 & 2: SentencePiece BPE, 32,000 vocab
- Uses Google's SentencePiece library
- BPE (Byte-Pair Encoding) algorithm
- 32,000 tokens covers most common English words and subwords
- Special tokens: `<s>` (begin), `</s>` (end), `<unk>` (unknown)
- Byte fallback: any character not in vocab is encoded as UTF-8 bytes

### LLaMA 3: tiktoken BPE, 128,000 vocab
- Switched to OpenAI's tiktoken library (same as GPT-4)
- 4× larger vocabulary: 128,000 tokens
- Better multilingual support
- More efficient encoding (fewer tokens per document = faster processing)
- The same English text takes ~30% fewer tokens compared to LLaMA 2

### Why vocabulary size matters:
- Larger vocab = each token covers more characters = shorter sequences = faster inference
- But larger vocab = larger embedding table = more memory
- 128K vocab for LLaMA 3 is a careful balance

### Example tokenization comparison:
```
Text: "What is the GSTIN number?"

LLaMA 2 (32K vocab): ["What", "▁is", "▁the", "▁G", "ST", "IN", "▁number", "?"] → 8 tokens
LLaMA 3 (128K vocab): ["What", "▁is", "▁the", "▁GSTIN", "▁number", "?"] → 6 tokens
```
(▁ = space prefix in SentencePiece)

---

## 5. Input Embeddings

After tokenization, each token ID is looked up in the embedding table:

```
token_id → embedding_table[token_id] → vector of shape [hidden_dim]
```

For LLaMA 2 7B: each token becomes a 4096-dimensional float vector.

The embeddings are **not normalized** — they're raw learned representations that the model training optimizes end-to-end.

**Weight tying:** The same embedding matrix is used both at the input (token → vector) and at the output (vector → vocabulary logits). This reduces parameters and improves training.

---

## 6. The Transformer Decoder Block

Each decoder block has two sub-layers, each wrapped with a residual connection:

```
x_new = x + Attention(RMSNorm(x))     # sub-layer 1
x_out = x_new + FFN(RMSNorm(x_new))   # sub-layer 2
```

### Why Residual Connections?
- Allows gradients to flow directly to earlier layers during backpropagation (avoids vanishing gradient problem)
- Lets layers learn incremental refinements ("add something useful to what already exists")
- Network can learn identity function (output = input) if a layer is not useful

### Why Pre-Norm (RMSNorm before attention/FFN)?
Original Transformers applied normalization after the sub-layer (Post-Norm). LLaMA uses Pre-Norm (normalize before):
- More stable training, especially at large scale
- Gradients don't blow up or vanish as easily
- Enables training without learning rate warmup tricks

---

## 7. Multi-Head Self-Attention (LLaMA 1 & 2)

Self-attention allows each token to "look at" all other previous tokens and determine what's relevant.

### Step-by-step attention computation:

**Step 1: Linear projections**
```python
Q = token_vectors @ W_Q    # shape: [seq_len, n_heads × head_dim]
K = token_vectors @ W_K    # shape: [seq_len, n_heads × head_dim]
V = token_vectors @ W_V    # shape: [seq_len, n_heads × head_dim]
```

For LLaMA 2 7B: n_heads = 32, head_dim = 128, so Q/K/V = [seq_len, 4096]

**Step 2: Reshape into heads**
```python
Q = reshape(Q, [seq_len, n_heads, head_dim])   # 32 separate query heads
K = reshape(K, [seq_len, n_heads, head_dim])   # 32 separate key heads
V = reshape(V, [seq_len, n_heads, head_dim])   # 32 separate value heads
```

**Step 3: Apply RoPE to Q and K** (see section 9)

**Step 4: Compute attention scores**
```python
scores = (Q @ K.transpose(-1,-2)) / sqrt(head_dim)
# shape: [seq_len, seq_len] per head
# scale by sqrt(128) = ~11.3 to prevent softmax saturation
```

**Step 5: Causal masking**
```python
# Mask future positions — token at position i cannot see position j > i
scores[i, j] = -infinity   if j > i
```

**Step 6: Softmax → attention weights**
```python
weights = softmax(scores, dim=-1)
# Each row sums to 1.0
# High weight = "pay a lot of attention to this token"
```

**Step 7: Weighted sum of values**
```python
attended = weights @ V    # shape: [seq_len, head_dim]
```

**Step 8: Concatenate all heads and project**
```python
output = concat(all heads) @ W_O   # shape: [seq_len, hidden_dim]
```

### Why Multiple Heads?
Each head learns to attend to different aspects:
- Head 1: syntactic relationships ("subject attends to verb")
- Head 2: coreference ("pronoun attends to noun it refers to")
- Head 3: positional proximity ("word attends to nearby words")
- etc.

---

## 8. Grouped Query Attention (LLaMA 2 70B & LLaMA 3)

GQA was introduced in LLaMA 2 for the 70B model and used for all sizes in LLaMA 3.

### The Memory Problem with Standard MHA
During inference, the KV cache stores K and V matrices for all previous tokens.  
For LLaMA 2 70B with full MHA:
- 80 attention heads × 2 (K and V) × 4096-char context × 8B float = enormous memory
- Makes it impossible to run on single GPU for long conversations

### GQA Solution
Group query heads to share K and V heads:

```
LLaMA 2 70B:
- Query heads: 64
- KV heads:    8   (groups of 8 queries share one K,V pair)

LLaMA 3 8B:
- Query heads: 32
- KV heads:    8   (groups of 4 queries share one K,V pair)

LLaMA 3 70B:
- Query heads: 64
- KV heads:    8   (groups of 8 queries share one K,V pair)
```

**Memory reduction:** 8× less KV cache for 70B model (8 KV heads vs 64)

**Quality tradeoff:** Very small — empirically GQA achieves ~99% of MHA quality at a fraction of the memory cost.

### LLaMA 1 & LLaMA 2 7B/13B/34B
These use standard MHA (all heads separate). GQA was only added to LLaMA 2 70B.
LLaMA 3 applied GQA to all model sizes.

---

## 9. Rotary Positional Embeddings (RoPE)

This is identical to Mistral's use of RoPE. See detailed explanation in the Mistral README, section 11.

### LLaMA-specific RoPE details:

**LLaMA 1:** `rope_theta = 10,000`  
**LLaMA 2:** `rope_theta = 10,000`  
**LLaMA 3:** `rope_theta = 500,000` (50× larger — allows much longer context)

### Why does rope_theta matter?
The theta value controls the rotation frequencies. Larger theta = lower maximum frequency = slower rotation = model can "remember" positions further apart. LLaMA 3's 500K theta is why it can handle 128K context with appropriate fine-tuning.

### RoPE Scaling Tricks
For LLaMA 3.1's 128K context, Meta uses **YaRN (Yet another RoPE extensioN)** scaling — a method to extend RoPE to longer contexts than the base training length by carefully adjusting rotation frequencies for different dimensions.

---

## 10. RMSNorm — Pre-Normalization

Same as used in Mistral. LLaMA was actually one of the first popular models to adopt RMSNorm.

```
RMSNorm(x) = (x / RMS(x)) × γ
where RMS(x) = sqrt(mean(x²) + ε)
```

- Only learnable parameter: γ (scale), shape `[hidden_dim]`
- No mean subtraction (unlike LayerNorm)
- No bias term β
- Applied before every attention and FFN sub-layer (Pre-Norm)

### LayerNorm vs RMSNorm (quick comparison):
```
LayerNorm: normalize by (mean, variance) — 2 statistics, 2 learned params (γ, β)
RMSNorm:   normalize by RMS only        — 1 statistic,  1 learned param (γ)
```
RMSNorm is about 10–15% faster at the same quality.

---

## 11. SwiGLU Feed-Forward Network

Same as Mistral. The FFN has 3 weight matrices (no biases):

```python
def ffn(x):
    gate   = x @ W_gate   # [seq, ffn_dim]
    up     = x @ W_up     # [seq, ffn_dim]
    hidden = silu(gate) * up    # element-wise gating
    out    = hidden @ W_down    # [seq, hidden_dim]
    return out

def silu(x):
    return x * sigmoid(x)   # smooth, non-saturating activation
```

### LLaMA FFN dimensions:

| Model | hidden_dim | ffn_dim |
|---|---|---|
| LLaMA 2 7B | 4096 | 11008 |
| LLaMA 2 13B | 5120 | 13824 |
| LLaMA 2 70B | 8192 | 28672 |
| LLaMA 3 8B | 4096 | 14336 |
| LLaMA 3 70B | 8192 | 28672 |

Note: LLaMA 3 8B has a larger FFN dim (14336) than LLaMA 2 7B (11008) — one reason it's stronger.

### Why SwiGLU over ReLU/GELU?
- ReLU: hard zero cutoff → dead neurons problem
- GELU: smooth approximation of ReLU → better but no gating
- SwiGLU: gating mechanism → each neuron can be selectively "opened" or "closed"
- Empirically, SwiGLU models consistently outperform ReLU/GELU at the same scale

---

## 12. KV Cache in LLaMA

During inference (text generation), LLaMA uses a **KV cache** to avoid recomputing attention for every previously generated token.

### Without KV cache:
To generate token 101, recompute K,V for tokens 1–100 every time = O(n²) computation.

### With KV cache:
- Token 1 processed: K₁, V₁ computed and cached
- Token 2 processed: K₂, V₂ computed, K₁V₁ loaded from cache
- Token N: only compute Kₙ, Vₙ. Load all previous from cache.
- Each new token: O(n) computation (reading cache) instead of O(n²)

### KV cache memory formula:
```
memory = 2 × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_element

LLaMA 2 7B (MHA, FP16):
= 2 × 32 × 32 × 128 × 4096 × 2 bytes
= ~2.1 GB for 4096 token context

LLaMA 3 8B (GQA, FP16):
= 2 × 32 × 8 × 128 × 8192 × 2 bytes
= ~1.07 GB for 8192 token context (GQA + longer context but less memory!)
```

This is why GQA matters — 4× fewer KV heads → 4× smaller cache → can serve longer contexts.

---

## 13. Context Length Evolution

| Model | Max Context | Typical Effective Context |
|---|---|---|
| LLaMA 1 7B | 2,048 tokens | ~2K |
| LLaMA 2 7B | 4,096 tokens | ~4K |
| LLaMA 3 8B | 8,192 tokens | ~8K |
| LLaMA 3.1 8B | 128,000 tokens | ~128K |

### What does "context" mean practically?
- 1 token ≈ 0.75 words ≈ 4 characters
- 4,096 tokens ≈ 3,000 words ≈ 6 pages of text
- 128,000 tokens ≈ 96,000 words ≈ a full novel

For invoice extraction in your project: most invoices are 500–2000 tokens, well within any model's context.

---

## 14. LLaMA 1 vs LLaMA 2 vs LLaMA 3 — Full Comparison

| Feature | LLaMA 1 | LLaMA 2 | LLaMA 3 |
|---|---|---|---|
| Release | Feb 2023 | Jul 2023 | Apr 2024 |
| Sizes | 7B, 13B, 33B, 65B | 7B, 13B, 34B, 70B | 8B, 70B, 405B |
| Context | 2,048 | 4,096 | 8,192 (128K for 3.1) |
| Tokenizer vocab | 32,000 | 32,000 | 128,000 |
| Tokenizer lib | SentencePiece | SentencePiece | tiktoken |
| Attention (small) | MHA | MHA | GQA |
| Attention (large) | MHA | GQA (70B only) | GQA |
| Normalization | RMSNorm | RMSNorm | RMSNorm |
| FFN activation | SwiGLU | SwiGLU | SwiGLU |
| Position encoding | RoPE (θ=10K) | RoPE (θ=10K) | RoPE (θ=500K) |
| RLHF (Chat) | No | Yes | Yes (improved) |
| Training tokens | 1T | 2T | 15T |
| License | Research only | Commercial (limited) | Open |
| Quality (8/7B) | GPT-3.5 level | Better | GPT-4 level on many tasks |

---

## 15. LLaMA vs Mistral — Architectural Differences

| Feature | LLaMA 2 7B | Mistral 7B |
|---|---|---|
| Attention | MHA (32 Q, 32 KV) | GQA (32 Q, 8 KV) |
| Sliding Window | No | Yes (W=4096) |
| KV Cache | Standard growing | Rolling buffer |
| Context length | 4,096 | 8,192 (SWA effective: ~131K) |
| FFN hidden dim | 11,008 | 14,336 |
| Vocab size | 32,000 | 32,000 |
| Normalization | RMSNorm | RMSNorm |
| Position | RoPE (θ=10K) | RoPE (θ=10K) |
| Key advantage | Established, well-studied | Faster, longer context, better quality |

Mistral was largely inspired by LLaMA's architecture and improved upon it.  
LLaMA 3 (April 2024) incorporated many of Mistral's improvements — GQA for all sizes, larger FFN, larger vocab.

---

## 16. Instruction Tuning — LLaMA-Chat / Instruct Models

Base LLaMA models (pretrained) just predict the next token — they don't "follow instructions." To make them useful assistants, Meta applies two stages of fine-tuning:

### Stage 1: Supervised Fine-Tuning (SFT)
- Collect (prompt, ideal response) pairs from human annotators
- Fine-tune the base model on this data
- Model learns the format of being a helpful assistant
- Result: model responds to instructions but may still be unsafe

### Stage 2: RLHF (see next section)

### Chat Template
LLaMA 2 Chat uses a specific prompt format:
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is the total amount on this invoice? [/INST]

The total amount is ₹1,299. </s>
```

LLaMA 3 uses a different template:
```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is the total amount?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

Using the wrong template gives poor results — the model expects a specific format it was trained on.

---

## 17. RLHF in LLaMA 2

**Reinforcement Learning from Human Feedback** is how LLaMA 2 Chat models are made safe and helpful.

### Step 1: Train a Reward Model
- Show humans pairs of model responses (A and B) to the same prompt
- Humans say which is better
- Train a separate model (reward model) to predict "how good is this response"
- The reward model learns: helpful + harmless + honest = high reward

### Step 2: PPO (Proximal Policy Optimization)
- The LLaMA model generates responses
- The reward model scores each response
- PPO updates LLaMA's weights to generate responses with higher reward
- A KL-divergence penalty prevents LLaMA from drifting too far from the SFT model (avoids reward hacking)

### Step 3: Iterative refinement
- Repeat: collect more preference data → train better reward model → PPO again
- LLaMA 2 went through multiple rounds

### LLaMA 2's specific RLHF innovations:
- **Two reward models:** One for helpfulness, one for safety (separate concerns)
- **Ghost Attention:** Technique to maintain system prompt instructions throughout long conversations
- **Multi-turn safety:** Trained specifically on multi-turn conversations, not just single turns

---

## 18. LLaMA 3 Specific Changes

LLaMA 3 made several key improvements over LLaMA 2:

### 1. Tokenizer: 32K → 128K vocabulary
- 4× more tokens
- Better multilingual coverage (non-Latin scripts particularly improved)
- More efficient encoding → fewer tokens per document → faster inference

### 2. GQA for all model sizes
- LLaMA 2 had GQA only for 70B; smaller models used full MHA
- LLaMA 3 uses GQA even for 8B (8 KV heads)
- Result: smaller KV cache, faster inference even for small models

### 3. Larger RoPE theta: 10K → 500K
- Enables much longer effective context
- LLaMA 3.1 with YaRN scaling reaches 128K context

### 4. Training data: 2T → 15T tokens
- 7.5× more training data
- Higher quality filtering (code, math, multilingual emphasized)
- The 8B model sees more data than the 70B LLaMA 2 model

### 5. Better instruction tuning
- Synthetic data generation (use LLaMA 3 to generate training data for itself)
- Preference optimization using DPO (Direct Preference Optimization) instead of full RLHF
- Result: 8B model beats LLaMA 2 70B on many benchmarks

### 6. Code and math emphasis
- LLaMA 3 training included more code and mathematical reasoning data
- Significant improvement on coding benchmarks (HumanEval, MBPP)
- Better at multi-step reasoning

---

## 19. How LLaMA Generates Text — Full Inference Flow

Complete walkthrough from prompt to generated text:

```
STEP 1: INPUT PREPARATION
User prompt: "What is the GSTIN on this invoice?"
→ Wrap in chat template
→ Tokenize: [128000, 128006, 9125, 128007, ...]
→ n tokens = 47 (for example)

STEP 2: PREFILL PHASE (process the entire prompt at once)
For each of the 32 decoder layers:
  a. Compute RMSNorm of current hidden states
  b. Compute Q, K, V projections for all 47 tokens simultaneously
  c. Apply RoPE rotations to Q and K
  d. Compute causal attention (token i sees tokens 0..i only)
  e. Cache all K and V values (shape: [47, n_kv_heads, head_dim])
  f. Project attention output, add residual
  g. RMSNorm
  h. SwiGLU FFN
  i. Add residual

After all 32 layers: final RMSNorm → linear → softmax
→ Sample next token (e.g., token 15 = "The")

STEP 3: DECODE PHASE (generate one token at a time)
For each new token:
  a. Embed the new token [1, hidden_dim]
  b. Pass through 32 layers, BUT:
     - Only compute Q for 1 new token
     - K, V for new token computed and APPENDED to cache
     - Attention reads from full cache (48 tokens now)
  c. Output next token

STEP 4: STOPPING
Continue until:
  - <|eot_id|> token generated (LLaMA 3 end token), OR
  - </s> token generated (LLaMA 2 end token), OR  
  - max_new_tokens reached

STEP 5: DETOKENIZE
[15, 3855, 27664, 374, 25] → "The GSTIN is 29ABCDE1234F1Z5"
```

### Key efficiency insight
The prefill (processing the prompt) is fast because all tokens are processed in parallel.
The decode phase is slow because tokens must be generated sequentially (each depends on the previous).
This is why LLaMA (and all autoregressive LLMs) are fast to start but slow to generate long responses.

---

## 20. Important Numbers to Remember

### LLaMA 2 7B

| Parameter | Value |
|---|---|
| Parameters | 6.7 billion |
| Layers | 32 |
| Hidden dim | 4,096 |
| FFN dim | 11,008 |
| Attention heads (Q) | 32 |
| KV heads | 32 (full MHA) |
| Head dim | 128 |
| Vocab size | 32,000 |
| Context length | 4,096 |
| RoPE theta | 10,000 |
| Training tokens | 2 trillion |

### LLaMA 3 8B

| Parameter | Value |
|---|---|
| Parameters | 8 billion |
| Layers | 32 |
| Hidden dim | 4,096 |
| FFN dim | 14,336 |
| Attention heads (Q) | 32 |
| KV heads | 8 (GQA) |
| Head dim | 128 |
| Vocab size | 128,000 |
| Context length | 8,192 |
| RoPE theta | 500,000 |
| Training tokens | 15 trillion |

### LLaMA 2 70B

| Parameter | Value |
|---|---|
| Parameters | 69.9 billion |
| Layers | 80 |
| Hidden dim | 8,192 |
| FFN dim | 28,672 |
| Attention heads (Q) | 64 |
| KV heads | 8 (GQA) |
| Head dim | 128 |

---

## 21. Common Interview Questions & Answers

**Q: What is LLaMA and why is it important?**  
A: LLaMA is Meta's family of open-source large language models. It's important because it made powerful LLMs freely available to researchers and developers — before LLaMA, the only strong models were GPT-3/4 which required paid API access. LLaMA sparked a wave of open-source models (Mistral, Falcon, Alpaca, Vicuna) that collectively built today's open AI ecosystem.

**Q: What is the difference between LLaMA 1, 2, and 3?**  
A: LLaMA 1 was research-only with 2K context; LLaMA 2 added RLHF safety training, GQA for the 70B model, and doubled context to 4K — with a commercial license; LLaMA 3 made a massive leap: 15× more training data, 128K vocabulary (vs 32K), GQA for all sizes, 500K RoPE theta for longer context, and the 8B model beats LLaMA 2 70B on benchmarks.

**Q: Explain how self-attention works.**  
A: Each token in the sequence creates three vectors — Query (what I'm looking for), Key (what I contain), and Value (what I contribute). Attention scores are computed as dot products of Query with all Keys, scaled and passed through softmax to get weights. The output is a weighted sum of all Values. This lets each token gather relevant information from any other token in the sequence. Multiple attention heads do this in parallel, each learning different relationships.

**Q: What is the KV cache and why is it needed?**  
A: The KV cache stores the Key and Value vectors computed for all previous tokens during inference. Without it, to generate each new token you'd have to recompute K and V for all previous tokens — O(n²) work for n tokens. With caching, you only compute K,V for the new token and load the rest from cache — O(n) work. This is what makes LLaMA practical for long conversations.

**Q: What is causal masking in the decoder?**  
A: In a decoder-only model, each token can only attend to itself and previous tokens — not future tokens. This is enforced by "causal masking" — setting attention scores for future positions to negative infinity before softmax, so they get zero weight. Without this, the model would "cheat" during training by looking at the answer it's supposed to predict.

**Q: Why does LLaMA use RMSNorm instead of LayerNorm?**  
A: RMSNorm is simpler (no mean subtraction, no bias) and about 10–15% faster, while achieving equivalent training stability. At LLaMA's scale, even small efficiency improvements compound significantly. LLaMA's success with RMSNorm was influential — most modern LLMs (Mistral, Falcon, Gemma) now also use RMSNorm.

**Q: What is instruction tuning and why is it needed?**  
A: A base (pretrained) LLaMA model just predicts the next token statistically — it doesn't understand that it should be helpful, follow instructions, or avoid harmful outputs. Instruction tuning fine-tunes the model on (instruction, ideal response) pairs using Supervised Fine-Tuning (SFT), then RLHF to align it with human preferences for helpfulness and safety. Without this, the model might complete your question rather than answer it.

**Q: What is RLHF? How does LLaMA 2 use it?**  
A: RLHF = Reinforcement Learning from Human Feedback. LLaMA 2 first trains a reward model on human preference data (which of two responses is better?). Then it uses PPO (Proximal Policy Optimization) to update LLaMA's weights to maximize the reward model's score, while a KL penalty prevents the model from deviating too far from the original. LLaMA 2 used two separate reward models — one for helpfulness, one for safety.

**Q: What is the difference between LLaMA and GPT?**  
A: Architecturally, they're very similar — both are decoder-only Transformers. The key differences are: LLaMA uses RoPE for position encoding (GPT-2/3 used learned or sinusoidal), LLaMA uses RMSNorm (GPT uses LayerNorm), LLaMA uses SwiGLU activation (GPT uses GELU), LLaMA has no bias in linear layers. The biggest practical difference is that LLaMA weights are open-source; GPT-3/4 weights are closed (API only).

**Q: How does LLaMA 3's 128K vocab help compared to 32K?**  
A: A larger vocabulary means each token represents more characters on average. For LLaMA 3, English text takes about 30% fewer tokens than LLaMA 2. Fewer tokens means: faster inference (fewer forward passes), longer effective context within the same context window, and better multilingual support (non-Latin languages need fewer byte-level fallback tokens). The tradeoff is a larger embedding matrix (128K × 4096 = 524M parameters just for embeddings).

**Q: What is the role of residual connections in LLaMA?**  
A: Residual connections add the layer's input directly to its output: `output = sub_layer(input) + input`. During backpropagation, gradients can flow directly through the addition without passing through the attention/FFN computation — this prevents vanishing gradients in deep networks (LLaMA has 32 layers). They also allow each layer to learn only the "correction" to add to what already exists, making training easier and results better.

---

*End of LLaMA Architecture Study Guide*
