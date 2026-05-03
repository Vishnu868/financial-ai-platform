# Open-Source LLM Models — Complete Reference Guide

> Who made them, what they are, how they work, where they run, and what they're best for.  
> Covers: LLaMA, Mistral, Qwen, Gemma, Phi, Falcon, DeepSeek, Command-R, and more.

---

## Table of Contents

1. [The Big Picture — How to Think About LLMs](#1-the-big-picture--how-to-think-about-llms)
2. [LLaMA (Meta)](#2-llama-meta)
3. [Mistral (Mistral AI)](#3-mistral-mistral-ai)
4. [Mixtral — Mixture of Experts (Mistral AI)](#4-mixtral--mixture-of-experts-mistral-ai)
5. [Qwen (Alibaba Cloud)](#5-qwen-alibaba-cloud)
6. [Gemma (Google DeepMind)](#6-gemma-google-deepmind)
7. [Phi (Microsoft)](#7-phi-microsoft)
8. [Falcon (Technology Innovation Institute)](#8-falcon-technology-innovation-institute)
9. [DeepSeek (DeepSeek AI)](#9-deepseek-deepseek-ai)
10. [Command-R (Cohere)](#10-command-r-cohere)
11. [Vicuna / Alpaca / OpenHermes — Fine-tunes](#11-vicuna--alpaca--openhermes--fine-tunes)
12. [Closed-Source Models for Comparison](#12-closed-source-models-for-comparison)
13. [Model Architecture Comparison Table](#13-model-architecture-comparison-table)
14. [Hardware Guide — CPU vs GPU](#14-hardware-guide--cpu-vs-gpu)
15. [Which Model for Which Device?](#15-which-model-for-which-device)
16. [Which Model for Which Task?](#16-which-model-for-which-task)
17. [Quantization — Making Models Smaller](#17-quantization--making-models-smaller)
18. [Running Models Locally — Tools](#18-running-models-locally--tools)
19. [Model Naming Conventions Explained](#19-model-naming-conventions-explained)
20. [Quick Selection Guide — Decision Tree](#20-quick-selection-guide--decision-tree)
21. [Common Interview Questions & Answers](#21-common-interview-questions--answers)

---

## 1. The Big Picture — How to Think About LLMs

All the models in this guide are **Large Language Models (LLMs)** — neural networks trained to predict the next token in a sequence. Despite coming from different companies, they share the same fundamental architecture: **decoder-only Transformer**.

### The key dimensions to compare any two models:

| Dimension | What it means |
|---|---|
| Parameters | Size of the model — more = smarter but slower and heavier |
| Architecture | Internal design choices (attention type, normalization, etc.) |
| Training data | What text was used and how much |
| Context length | How many tokens it can read at once |
| License | Can you use it commercially? |
| Quantization | Can it be compressed to run on consumer hardware? |
| Specialization | Is it general-purpose, code, math, multimodal, or language-specific? |

### The Closed vs Open Divide

| Closed (API only) | Open (download weights) |
|---|---|
| GPT-4, GPT-4o (OpenAI) | LLaMA 3 (Meta) |
| Claude 3 (Anthropic) | Mistral 7B (Mistral AI) |
| Gemini 1.5 (Google) | Qwen2.5 (Alibaba) |
| Command+ (Cohere) | Gemma 2 (Google) |

Open models let you run everything locally — no API costs, no data leaving your machine. This is why they're important for production deployments and privacy-sensitive applications like financial documents.

---

## 2. LLaMA (Meta)

### Who made it?
**Meta AI** (Facebook's parent company). Research team led by Guillaume Lample and others.

### The family
| Model | Release | Sizes | Context | Key feature |
|---|---|---|---|---|
| LLaMA 1 | Feb 2023 | 7B, 13B, 33B, 65B | 2K | First open LLM to compete with GPT-3 |
| LLaMA 2 | Jul 2023 | 7B, 13B, 34B, 70B | 4K | RLHF safety, commercial license |
| LLaMA 3 | Apr 2024 | 8B, 70B | 8K | 15T training tokens, 128K vocab |
| LLaMA 3.1 | Jul 2024 | 8B, 70B, 405B | 128K | Long context, multilingual |
| LLaMA 3.2 | Sep 2024 | 1B, 3B, 11B, 90B | 128K | Vision models (multimodal) |
| LLaMA 3.3 | Dec 2024 | 70B | 128K | Improved instruct fine-tune |

### Architecture highlights
- Decoder-only Transformer
- RMSNorm (pre-normalization)
- SwiGLU activation
- RoPE positional encoding
- GQA (Grouped Query Attention) from LLaMA 3 onwards

### License
- LLaMA 1: Research only (leaked, then released)
- LLaMA 2: Free for most commercial use (restrictions for >700M monthly users)
- LLaMA 3: Llama 3 Community License — essentially open for most uses

### Strengths
- Best general-purpose open model at its scale
- Massive ecosystem (fine-tunes, quantizations, tools built on top)
- Excellent instruction following in instruct variants
- Strong code generation
- Great multilingual support in LLaMA 3.1+

### Weaknesses
- LLaMA 2 7B is weaker than Mistral 7B on most tasks
- Requires more VRAM than Mistral due to MHA (LLaMA 2)
- 70B needs high-end hardware (GPU or very fast CPU)

### Best use cases
- General Q&A and chat
- Summarization
- Code generation (LLaMA 3)
- Instruction following
- Building fine-tunes (huge community support)
- RAG pipelines (excellent context understanding)

### Where it runs
- LLaMA 3 8B: Runs on any 8GB VRAM GPU (quantized), or 16GB RAM CPU (slow)
- LLaMA 3 70B: Needs 40GB+ VRAM (A100/H100) or 64GB RAM (CPU, very slow)
- LLaMA 3.2 1B/3B: Runs on mobile devices (Meta even ships these in apps)

---

## 3. Mistral (Mistral AI)

### Who made it?
**Mistral AI** — a French startup founded in 2023 by ex-DeepMind and ex-Meta researchers (Arthur Mensch, Guillaume Lample, Timothée Lacroix).

### The family
| Model | Release | Size | Context | Key feature |
|---|---|---|---|---|
| Mistral 7B | Sep 2023 | 7.3B | 8K | GQA + SWA — beats LLaMA 2 13B |
| Mistral 7B Instruct | Sep 2023 | 7.3B | 8K | Instruction-tuned version |
| Mixtral 8×7B | Dec 2023 | 46.7B total / 13B active | 32K | Mixture of Experts |
| Mistral Small | Feb 2024 | ~22B | 32K | Enterprise, API |
| Mistral Large | Feb 2024 | ~123B | 128K | Flagship, API only |
| Codestral | May 2024 | 22B | 32K | Code-specific |
| Mistral NeMo | Jul 2024 | 12B | 128K | Joint with NVIDIA |
| Mistral 7B v0.3 | Jun 2024 | 7.3B | 32K | Improved instruct |

### Architecture highlights (Mistral 7B)
- Grouped Query Attention (32 Q heads, 8 KV heads)
- Sliding Window Attention (W = 4096)
- Rolling Buffer KV Cache
- RMSNorm, SwiGLU, RoPE
- No bias in linear layers

### License
- Mistral 7B: Apache 2.0 — fully open, commercial use, no restrictions
- Mistral Large: API only, not open weights

### Strengths
- Best-in-class quality at 7B parameters
- Very memory-efficient (GQA + rolling cache)
- Fastest inference at its size class
- Apache 2.0 = maximum freedom
- Strong instruction following
- Good for financial/structured document tasks

### Weaknesses
- Smaller community than LLaMA
- Less fine-tune variety
- Mistral Large and other advanced models are API-only

### Best use cases
- **Your project:** RAG chatbot for invoice Q&A (Ollama + Mistral)
- Any CPU/low-VRAM deployment (most memory-efficient at 7B)
- Structured extraction from documents
- Instruction following
- Chatbots that need to run locally

### Where it runs
- Mistral 7B quantized (Q4): ~4GB RAM → runs on any modern laptop CPU
- Mistral 7B quantized (Q8): ~7GB VRAM → runs on RTX 3060/4060
- Mistral 7B full precision: 14GB VRAM → A100 / RTX 4090
- **Ollama:** `ollama pull mistral` — easiest local setup, used in your project

---

## 4. Mixtral — Mixture of Experts (Mistral AI)

### What is it?
Mixtral 8×7B is an extension of Mistral using **Mixture of Experts (MoE)** architecture.

### How MoE works
Instead of one Feed-Forward Network (FFN) in each Transformer block, there are **8 parallel FFN "experts"**. A **router network** looks at each token and selects the top 2 experts to process it.

```
Token arrives at layer →
Router: "This token is about finance" →
Activates Expert 3 (financial reasoning) + Expert 7 (general language)
Output = 0.6 × Expert3(token) + 0.4 × Expert7(token)
```

### The numbers
- Total parameters: 8 experts × 7B = ~46.7B
- Active parameters per token: 2 experts × 7B = ~13B
- Result: quality of a 46B model, compute cost of a 13B model

### Memory requirements
- Despite 46.7B total params, you only need enough VRAM to hold all 46.7B (not just the active 13B)
- Q4 quantized: ~26GB → needs two GPUs or a GPU with 32GB VRAM
- CPU: needs ~50GB RAM

### Mixtral 8×22B
Released later, larger MoE:
- 8 experts × 22B each = ~141B total, ~39B active
- Context: 64K tokens
- Stronger than Mixtral 8×7B, needs ~45GB VRAM quantized

### Strengths
- GPT-3.5 quality at much lower compute cost
- Excellent at multi-domain reasoning
- Strong code generation
- 32K context length

### Weaknesses
- High memory even quantized (need 24GB+ VRAM)
- More complex to serve than dense models
- Not ideal for pure CPU inference

---

## 5. Qwen (Alibaba Cloud)

### Who made it?
**Alibaba Cloud's Tongyi Laboratory** — Alibaba's AI research division based in China.

### The family
| Model | Release | Sizes | Context | Key feature |
|---|---|---|---|---|
| Qwen 1.5 | Feb 2024 | 0.5B–110B | 32K | Multiple sizes, multilingual |
| Qwen 2 | Jun 2024 | 0.5B–72B | 128K | Strong English + Chinese |
| Qwen 2.5 | Sep 2024 | 0.5B–72B | 128K | Best open model at 72B |
| Qwen 2.5-Coder | Sep 2024 | 1.5B–32B | 128K | Code-specialized |
| Qwen 2.5-Math | Sep 2024 | 1.5B–72B | 4K | Math-specialized |
| QwQ-32B | Nov 2024 | 32B | 32K | Reasoning (like o1) |
| Qwen-VL | Ongoing | 7B–72B | 32K | Vision-language multimodal |

### Architecture highlights
- Decoder-only Transformer
- GQA (Grouped Query Attention)
- RoPE with YaRN for long context
- RMSNorm
- SwiGLU
- Tied embeddings

### License
- Qwen 2.5 72B and below: **Qwen License** (mostly open, some restrictions for large commercial deployments)
- Qwen 2.5 72B Instruct: Generally considered open enough for most uses

### Strengths
- **Exceptional multilingual:** Strongest open model for Chinese + English
- **Qwen 2.5 72B beats LLaMA 3 70B** on many benchmarks
- Wide range of sizes (0.5B to 72B) — something for every device
- Excellent coding models (Qwen 2.5-Coder)
- Strong math reasoning (Qwen 2.5-Math)
- **QwQ-32B:** One of the best reasoning models (chain-of-thought, matches o1-mini)
- Great for Indian businesses — handles mixed English/Hindi/other scripts well

### Weaknesses
- Chinese company → some distrust in Western enterprise deployments
- License has commercial deployment size restrictions
- Community smaller than LLaMA
- Qwen-VL multimodal still maturing

### Best use cases
- Multilingual applications (Chinese, Japanese, Korean, Arabic + English)
- Indian startups needing multilingual support
- Coding assistants (Qwen 2.5-Coder)
- Math and reasoning tasks
- When you need the strongest possible open model (72B)

### Where it runs
- Qwen 2.5 0.5B: Runs on mobile (500M params)
- Qwen 2.5 7B: Any 8GB GPU or 16GB RAM laptop
- Qwen 2.5 32B: 24GB VRAM GPU or 64GB RAM server
- Qwen 2.5 72B: 48GB VRAM (2×24GB GPUs) or 128GB RAM server

---

## 6. Gemma (Google DeepMind)

### Who made it?
**Google DeepMind** — combination of Google Brain and DeepMind research labs.

### The family
| Model | Release | Sizes | Context | Key feature |
|---|---|---|---|---|
| Gemma 1 | Feb 2024 | 2B, 7B | 8K | Google's first open model |
| Gemma 2 | Jun 2024 | 2B, 9B, 27B | 8K | Beats models 2× its size |
| CodeGemma | Apr 2024 | 2B, 7B | 8K | Code specialized |
| RecurrentGemma | Apr 2024 | 2B | 8K | Experimental recurrent architecture |
| PaliGemma | May 2024 | 3B | — | Vision-language multimodal |
| Gemma 2 9B | Jun 2024 | 9B | 8K | Best 9B model available |

### Architecture highlights (Gemma 2 — unique features)
- **Multi-Query Attention for some layers, GQA for others** (alternating pattern)
- **Sliding Window Attention** (alternating: some layers global, some windowed)
- **Logit soft-capping:** Prevents logits from exploding (clips at ±50)
- **Post-normalization:** Applies RMSNorm AFTER attention/FFN in addition to pre-norm
- RoPE, SwiGLU, RMSNorm

### License
- Gemma License — allows commercial use but has some restrictions (no training other Gemma models, branding requirements)

### Strengths
- **Gemma 2 9B beats LLaMA 3 70B** on many benchmarks — extraordinary efficiency
- Google's training data quality is very high
- Excellent reasoning and instruction following
- Good safety training
- PaliGemma is strong for vision-language tasks

### Weaknesses
- Context only 8K (shorter than Mistral, Qwen, LLaMA 3)
- License more restrictive than Apache 2.0
- Not as good for code as Qwen 2.5-Coder
- Smaller community than LLaMA

### Best use cases
- When you need maximum quality at small model size
- Safety-critical applications (strong safety training from Google)
- Research and academic use
- Vision tasks (PaliGemma)

### Where it runs
- Gemma 2 2B: Runs on mobile or very basic CPU (2B params)
- Gemma 2 9B: 8GB VRAM GPU or 16GB RAM laptop
- Gemma 2 27B: 24GB VRAM or 48GB RAM

---

## 7. Phi (Microsoft)

### Who made it?
**Microsoft Research** — specifically the team led by Sebastien Bubeck.

### The family
| Model | Release | Size | Context | Key feature |
|---|---|---|---|---|
| Phi-1 | Jun 2023 | 1.3B | 2K | "Textbooks are all you need" |
| Phi-1.5 | Sep 2023 | 1.3B | 2K | Common sense reasoning |
| Phi-2 | Dec 2023 | 2.7B | 2K | Best 2.7B model at launch |
| Phi-3-mini | Apr 2024 | 3.8B | 4K/128K | Matches Mistral 7B |
| Phi-3-small | Apr 2024 | 7B | 8K/128K | Multilingual, strong |
| Phi-3-medium | Apr 2024 | 14B | 4K/128K | Near-GPT-3.5 quality |
| Phi-3.5-mini | Aug 2024 | 3.8B | 128K | Long context mini model |
| Phi-3.5-MoE | Aug 2024 | 41B total/6.6B active | 128K | MoE version |
| Phi-4 | Dec 2024 | 14B | 16K | Strongest Phi yet |

### Core philosophy — "Small Language Models"
Microsoft's insight: **quality of training data matters more than quantity**. They trained Phi on:
- Carefully filtered, high-quality web text
- "Textbook-quality" synthetic data generated by GPT-4
- Code and math datasets
- Result: 3.8B Phi-3-mini matches or beats Mistral 7B on many benchmarks

### Architecture highlights
- Standard decoder-only Transformer
- Full multi-head attention (smaller models don't need GQA)
- RoPE, LayerNorm (NOT RMSNorm — unusual)
- GELU activation (not SwiGLU)
- Flash Attention support

### License
- MIT License — most permissive possible, fully open commercial use

### Strengths
- **Best small model:** Phi-3-mini (3.8B) outperforms many 7B models
- **MIT license** — truly free for everything
- Excellent for mobile/edge deployment
- Strong reasoning relative to size
- Good coding ability
- Phi-4 (14B) is near state-of-the-art for its size class

### Weaknesses
- Short context in early versions (2K/4K)
- Not as strong for creative writing
- Less multilingual than Qwen
- Smaller community

### Best use cases
- **Mobile applications** (3.8B runs on high-end phones)
- **Edge devices** (Raspberry Pi, Jetson Nano with smaller versions)
- **Low-memory servers** where every GB counts
- **Code assistance** at small scale
- **Embedded applications** (MIT license, no restrictions)

### Where it runs
- Phi-3-mini 3.8B: 2-4GB RAM — runs on phones, Raspberry Pi 5, cheap laptops
- Phi-3-small 7B: 8GB RAM or 6GB VRAM
- Phi-3-medium 14B: 16GB RAM or 10GB VRAM
- Phi-4 14B: 12GB VRAM or 24GB RAM

---

## 8. Falcon (Technology Innovation Institute)

### Who made it?
**Technology Innovation Institute (TII)** — a research center in Abu Dhabi, UAE.

### The family
| Model | Release | Sizes | Context | Key feature |
|---|---|---|---|---|
| Falcon 1 | May 2023 | 7B, 40B | 2K | First strong open model |
| Falcon 1 (180B) | Sep 2023 | 180B | 2K | Largest open model at release |
| Falcon 2 | May 2024 | 11B | 8K | Improved, multimodal variant |
| Falcon 2 VLM | May 2024 | 11B | 8K | Vision-language |

### Architecture highlights
- Multi-Query Attention (MQA) — one single K,V head (more aggressive than GQA)
- Parallel attention + FFN (compute simultaneously, not sequentially)
- ALiBi positional encoding (not RoPE — encodes positions via attention bias)
- Custom CUDA kernels for efficiency

### License
- Falcon 180B: Research only
- Falcon 7B, 40B, 11B: Apache 2.0 (commercial use allowed)

### Strengths
- Falcon 40B was the best open model in early 2023
- Efficient inference (MQA + parallel layers)
- Strong English-language performance
- Good for long documents with ALiBi

### Weaknesses
- **Largely surpassed** by LLaMA 3, Mistral, and Qwen in 2024
- ALiBi context extrapolation is less predictable than RoPE
- Smaller active community now
- Less instruction-tuning variety

### Best use cases
- Legacy deployments already using Falcon
- Research comparisons
- When Apache 2.0 license is required and LLaMA/Mistral aren't available

---

## 9. DeepSeek (DeepSeek AI)

### Who made it?
**DeepSeek** — a Chinese AI company founded by High-Flyer Capital Management (a quantitative hedge fund). The team is known for very compute-efficient training.

### The family
| Model | Release | Sizes | Context | Key feature |
|---|---|---|---|---|
| DeepSeek LLM | Nov 2023 | 7B, 67B | 4K | Strong base model |
| DeepSeek Coder | Nov 2023 | 1.3B–33B | 16K | Top coding model |
| DeepSeek MoE | Jan 2024 | 16B total / 2.8B active | 4K | Ultra-efficient MoE |
| DeepSeek V2 | May 2024 | 236B total / 21B active | 128K | Multi-head Latent Attention |
| DeepSeek Coder V2 | Jun 2024 | 236B/21B | 128K | Best open coding model |
| DeepSeek V3 | Dec 2024 | 671B total / 37B active | 128K | Near GPT-4o quality |
| **DeepSeek R1** | Jan 2025 | 671B total / 37B active | 128K | **Chain-of-thought reasoning, shocked the world** |
| DeepSeek R1 Distill | Jan 2025 | 1.5B–70B | 128K | Smaller R1 variants |

### Architecture highlights (DeepSeek V2/V3)
DeepSeek V2 introduced a revolutionary attention mechanism: **Multi-head Latent Attention (MLA)**

Standard attention stores K and V for every head → huge KV cache.  
MLA compresses K,V into a **low-rank latent vector** then decompresses at attention time.

```
Standard: K ∈ [seq, n_heads × head_dim]  → huge cache
MLA:      C_KV ∈ [seq, latent_dim]       → tiny cache (latent_dim << n_heads × head_dim)
          Then: K, V = decompress(C_KV) at query time
```

Result: 93.3% reduction in KV cache size for DeepSeek V2.

### DeepSeek R1 — Why It Shocked the Industry
Released January 2025, R1 matched or beat OpenAI o1 (their best reasoning model) at a fraction of the cost. It achieved this with:
- **Pure reinforcement learning** — no supervised fine-tuning, just RL on math/code problems
- **Chain-of-thought reasoning** that emerges naturally from RL training
- Trained for a fraction of GPT-4/o1's compute cost
- Open-source release of both the full model and distilled smaller versions

### License
- DeepSeek models: MIT License (most are fully open)
- R1 distilled models: MIT License

### Strengths
- **DeepSeek R1** is the best open-source reasoning model available
- **DeepSeek Coder** is among the best coding models
- Exceptional compute efficiency (strong models trained cheaply)
- MIT license — fully open
- Distilled R1 models (7B, 14B, 32B) bring reasoning to smaller hardware

### Weaknesses
- Chinese company → compliance concerns in some enterprises
- Very large flagship models (671B) not practical for local deployment
- Newer ecosystem, fewer integrations than LLaMA

### Best use cases
- **Complex reasoning tasks** (math, multi-step problems) → R1 or R1-Distill
- **Coding** → DeepSeek Coder V2
- **When you need GPT-4-level quality open-source**
- Research and benchmarking

### Where it runs
- R1 Distill 7B: 8GB GPU or 16GB RAM — very practical
- R1 Distill 14B: 12GB VRAM or 24GB RAM
- R1 Distill 32B: 24GB VRAM or 64GB RAM
- Full R1 671B: Multiple A100/H100 GPUs — data center only

---

## 10. Command-R (Cohere)

### Who made it?
**Cohere** — a Canadian AI company founded by ex-Google Brain researchers, focused on enterprise NLP.

### The family
| Model | Release | Size | Context | Key feature |
|---|---|---|---|---|
| Command-R | Mar 2024 | 35B | 128K | RAG-optimized |
| Command-R+ | Apr 2024 | 104B | 128K | Enterprise flagship |
| Aya | Feb 2024 | 8B, 35B | 8K | 101-language multilingual |
| Command-R7B | Nov 2024 | 7B | 128K | Efficient RAG model |

### Architecture highlights
- Decoder-only Transformer
- GQA (Grouped Query Attention)
- RoPE positional encoding
- **Trained specifically for RAG** — model understands "citation" and "groundedness"
- Multi-hop retrieval capabilities

### License
- Command-R (35B): CC-BY-NC (non-commercial only for full model)
- Command-R7B: Research use
- Cohere API: Commercial (paid)

### Strengths
- **Best RAG performance** — trained explicitly for retrieval-augmented generation
- 128K context — can read entire documents
- Grounded responses with citations
- Excellent multilingual (Aya models)
- Strong structured output

### Weaknesses
- Commercial use restricted without API
- Less general-purpose than LLaMA
- Smaller community

### Best use cases
- **Production RAG pipelines** — this is what it's built for
- Document Q&A systems
- Knowledge base chatbots
- Enterprise search augmentation

---

## 11. Vicuna / Alpaca / OpenHermes — Fine-tunes

These are not original models — they are **fine-tunes** built on top of LLaMA, Mistral, etc.

| Model | Base | Who made it | What it is |
|---|---|---|---|
| Alpaca | LLaMA 1 | Stanford | First instruction-tuned LLaMA, used GPT-4 to generate training data |
| Vicuna | LLaMA | LMSYS | Fine-tuned on ShareGPT conversations |
| OpenHermes 2.5 | Mistral 7B | NousResearch | Strong general instruct fine-tune |
| Nous Hermes | LLaMA 2/3 | NousResearch | Popular general-purpose fine-tune |
| WizardLM | LLaMA | Microsoft | Evolved Instructions training method |
| Dolphin | LLaMA/Mistral | Eric Hartford | Uncensored assistant |
| DeepSeek R1 Distill | Qwen/LLaMA 3 | DeepSeek | R1 reasoning distilled into smaller models |
| Llama-3-Groq | LLaMA 3 | Groq | Optimized for Groq inference chips |

### Fine-tuning methods used:
- **SFT (Supervised Fine-Tuning):** Train on (instruction, response) pairs
- **RLHF (Reinforcement Learning from Human Feedback):** Human preference data + PPO
- **DPO (Direct Preference Optimization):** Simpler alternative to RLHF, no reward model needed
- **LoRA / QLoRA:** Fine-tune only a small subset of parameters (efficient, popular for custom fine-tunes)

---

## 12. Closed-Source Models for Comparison

Understanding where open models stand relative to the best closed models:

| Model | Company | Best at | Roughly equivalent open model |
|---|---|---|---|
| GPT-4o | OpenAI | General, multimodal | Llama 3.1 405B (close but not quite) |
| GPT-4o mini | OpenAI | Efficient GPT-4 | LLaMA 3.1 70B |
| o1 / o3 | OpenAI | Reasoning | DeepSeek R1 |
| Claude 3.5 Sonnet | Anthropic | Coding, instruction | LLaMA 3.1 405B (coding close) |
| Gemini 1.5 Pro | Google | Long context, multimodal | No direct equivalent yet |
| Gemini 1.5 Flash | Google | Fast, efficient | LLaMA 3.1 8B |

**Key insight:** As of early 2025, the best open-source models (DeepSeek R1, LLaMA 3.1 405B) have essentially caught up with GPT-4-class performance on many benchmarks. The gap is closing rapidly.

---

## 13. Model Architecture Comparison Table

| Model | Params | Layers | Hidden dim | Attention | FFN act | Norm | Context | Vocab |
|---|---|---|---|---|---|---|---|---|
| LLaMA 2 7B | 6.7B | 32 | 4096 | MHA (32/32) | SwiGLU | RMSNorm | 4K | 32K |
| LLaMA 3 8B | 8B | 32 | 4096 | GQA (32/8) | SwiGLU | RMSNorm | 8K | 128K |
| LLaMA 3.1 70B | 70B | 80 | 8192 | GQA (64/8) | SwiGLU | RMSNorm | 128K | 128K |
| Mistral 7B | 7.3B | 32 | 4096 | GQA+SWA (32/8) | SwiGLU | RMSNorm | 8K | 32K |
| Mixtral 8×7B | 46.7B | 32 | 4096 | GQA+SWA (32/8) | SwiGLU | RMSNorm | 32K | 32K |
| Qwen 2.5 7B | 7.6B | 28 | 3584 | GQA (28/4) | SwiGLU | RMSNorm | 128K | 152K |
| Qwen 2.5 72B | 72.7B | 80 | 8192 | GQA (64/8) | SwiGLU | RMSNorm | 128K | 152K |
| Gemma 2 9B | 9.2B | 42 | 3584 | GQA + sliding | SwiGLU | RMSNorm | 8K | 256K |
| Gemma 2 27B | 27.2B | 46 | 4608 | GQA + sliding | SwiGLU | RMSNorm | 8K | 256K |
| Phi-3-mini | 3.8B | 32 | 3072 | MHA (32/32) | GELU | LayerNorm | 128K | 32K |
| Phi-4 | 14B | 40 | 5120 | GQA | SwiGLU | LayerNorm | 16K | 100K |
| Falcon 2 11B | 11B | 60 | 4096 | GQA (32/8) | GELU | LayerNorm | 8K | 65K |
| DeepSeek V3 | 671B/37B | 61 | 7168 | MLA | SwiGLU | RMSNorm | 128K | 129K |
| DeepSeek R1 | 671B/37B | 61 | 7168 | MLA | SwiGLU | RMSNorm | 128K | 129K |
| Command-R | 35B | — | — | GQA | SwiGLU | RMSNorm | 128K | — |

---

## 14. Hardware Guide — CPU vs GPU

### GPU Inference

A **GPU (Graphics Processing Unit)** has thousands of small cores optimized for parallel matrix operations — exactly what Transformer attention and FFN layers need.

**Why GPU is preferred:**
- Matrix multiply (the core of attention) is massively parallel
- Modern GPUs have dedicated **Tensor Cores** for FP16/INT8 matrix ops
- High memory bandwidth: A100 = 2TB/s, RTX 4090 = 1TB/s
- Can process entire batches simultaneously

**GPU types for LLMs:**

| GPU | VRAM | Approx cost | Best for |
|---|---|---|---|
| RTX 3060 | 12GB | ₹25,000 | Mistral 7B, Phi-3, Gemma 9B (quantized) |
| RTX 3090/4090 | 24GB | ₹80,000–1,50,000 | Mistral 7B full, Mixtral 8×7B (Q4), Qwen 32B |
| RTX 4070 Ti | 12GB | ₹55,000 | Same as 3060 but faster |
| RTX 4080 | 16GB | ₹80,000 | LLaMA 3 8B full, 13B quantized |
| A100 40GB | 40GB | ₹8,00,000+ | LLaMA 70B, Mixtral 8×7B full |
| A100 80GB | 80GB | ₹15,00,000+ | LLaMA 70B full, DeepSeek V3 (1 GPU) |
| 2× A100 80GB | 160GB | ₹30,00,000+ | DeepSeek R1 / V3, LLaMA 405B |

**Apple Silicon (M1/M2/M3):**
Apple's chips have unified memory — the GPU and CPU share the same RAM pool. An M2 Max with 96GB unified memory can run a 70B model at decent speed. Better than CPU-only, competitive with mid-range GPUs.

### CPU Inference

CPU inference is slow but accessible — any computer with enough RAM can run a quantized model.

**Why CPU is slow:**
- Fewer compute cores (16 vs 10,000+ on GPU)
- Lower memory bandwidth (DDR5: ~100GB/s vs GPU: 1TB/s — 10× slower)
- Sequential processing of many operations

**CPU performance comparison for Mistral 7B Q4:**

| CPU | RAM | Speed (tokens/sec) | Verdict |
|---|---|---|---|
| Intel i5-12th gen | 16GB | 3–5 tok/s | Usable for dev, slow for production |
| Intel i9-13th gen | 32GB | 6–10 tok/s | Reasonable |
| AMD Ryzen 9 7950X | 64GB | 15–20 tok/s | Good for 7B models |
| Apple M2 Pro | 32GB | 20–30 tok/s | Excellent for CPU inference |
| Apple M3 Max | 128GB | 40–60 tok/s | Can run 70B models at usable speed |
| AMD Threadripper | 128GB | 20–30 tok/s | Can run 70B quantized |

**For your laptop (typical 16GB RAM, Intel/AMD):**
- Phi-3-mini 3.8B (Q4): 10–15 tok/s — comfortable
- Mistral 7B (Q4): 3–7 tok/s — slow but works
- LLaMA 3 8B (Q4): 3–6 tok/s — slow

---

## 15. Which Model for Which Device?

### Mobile Phone (4–8GB RAM)
| Model | Size | Notes |
|---|---|---|
| Phi-3-mini Q4 | 3.8B / ~2GB | Best choice — MIT license, strong quality |
| LLaMA 3.2 1B | 1B / ~0.7GB | Fastest, very basic quality |
| LLaMA 3.2 3B | 3B / ~1.8GB | Good balance |
| Gemma 2 2B | 2B / ~1.2GB | Google quality, small |
| Qwen 2.5 0.5B/1.5B | 0.5–1.5B | Good multilingual |

**Framework:** llama.cpp, MLC-LLM, or Ollama Edge

### Laptop — 8GB RAM (no dedicated GPU or integrated GPU)
| Model | Quantization | Speed | Quality |
|---|---|---|---|
| Phi-3-mini 3.8B | Q4_K_M | 8–12 tok/s | Good |
| Gemma 2 2B | Q8 | 10–15 tok/s | Good |
| LLaMA 3.2 3B | Q4 | 8–12 tok/s | Good |
| Mistral 7B | Q4 | 3–5 tok/s | Better quality, slow |

**Recommendation:** Phi-3-mini for daily use. Mistral 7B if you need better quality and can wait.

### Laptop/Desktop — 16GB RAM
| Model | Quantization | Speed | Quality |
|---|---|---|---|
| Mistral 7B | Q4_K_M | 5–8 tok/s | Excellent for 7B |
| LLaMA 3 8B | Q4_K_M | 5–8 tok/s | Strong |
| Qwen 2.5 7B | Q4_K_M | 5–8 tok/s | Strong multilingual |
| Gemma 2 9B | Q4 | 4–6 tok/s | Outstanding quality |
| Phi-3-medium 14B | Q4 | 2–4 tok/s | Near GPT-3.5 |

**Recommendation:** Mistral 7B (Apache 2.0) or Gemma 2 9B (outstanding quality).

### Workstation — 32GB RAM
| Model | Quantization | Speed | Quality |
|---|---|---|---|
| LLaMA 3 8B | Q8 (full quality) | 8–12 tok/s | Strong |
| Mistral 7B | FP16 (full quality) | 6–10 tok/s | Strong |
| Phi-3-medium 14B | Q4_K_M | 5–8 tok/s | Near GPT-3.5 |
| Qwen 2.5 14B | Q4_K_M | 4–7 tok/s | Very strong |
| DeepSeek R1 Distill 14B | Q4 | 3–5 tok/s | Excellent reasoning |

### Server — 64GB RAM
| Model | Quantization | Speed | Quality |
|---|---|---|---|
| LLaMA 3 70B | Q4 | 3–6 tok/s | Near GPT-4 class |
| Qwen 2.5 72B | Q4 | 3–6 tok/s | Best open model |
| DeepSeek R1 Distill 32B | Q4 | 5–8 tok/s | Best reasoning at this size |
| Mixtral 8×7B | Q4 | 3–5 tok/s | GPT-3.5 quality |

### GPU Setup — 8GB VRAM (RTX 3060 / RTX 4060)
| Model | Quantization | Speed | Quality |
|---|---|---|---|
| Mistral 7B | Q4_K_M | 30–50 tok/s | Excellent |
| LLaMA 3 8B | Q4_K_M | 25–40 tok/s | Strong |
| Gemma 2 9B | Q4 | 20–35 tok/s | Outstanding |
| Phi-3-mini | Q8 | 50–80 tok/s | Good |
| Qwen 2.5 7B | Q4 | 30–50 tok/s | Strong |

**GPU is 8–10× faster than CPU for the same model size.**

### GPU Setup — 24GB VRAM (RTX 3090 / RTX 4090)
| Model | Quantization | Speed | Quality |
|---|---|---|---|
| LLaMA 3 8B | FP16 full | 60–90 tok/s | Full quality |
| Mistral 7B | FP16 full | 70–100 tok/s | Full quality |
| Qwen 2.5 32B | Q4 | 15–25 tok/s | Very strong |
| Mixtral 8×7B | Q4 | 20–30 tok/s | GPT-3.5 quality |
| DeepSeek R1 Distill 14B | Q8 | 30–50 tok/s | Excellent reasoning |

### For Your Project (Financial Document AI)
Currently using Mistral 7B via Ollama on CPU. 

**Best upgrade path:**
1. **Stay on CPU:** Switch to `phi3:mini` for faster responses or `qwen2.5:7b` for better multilingual
2. **Add a GPU (RTX 3060):** Mistral 7B goes from 5 tok/s → 45 tok/s — 9× speedup
3. **Better model for RAG:** `command-r:7b` (specifically built for RAG) or `llama3.1:8b`

---

## 16. Which Model for Which Task?

### General Q&A / Chat
1. LLaMA 3.1 8B Instruct (balanced quality + size)
2. Mistral 7B Instruct (fast, Apache 2.0)
3. Qwen 2.5 7B Instruct (multilingual)

### Coding / Programming
1. DeepSeek Coder V2 (best open coding model)
2. Qwen 2.5-Coder 7B (excellent, runs locally)
3. LLaMA 3 8B (good general coding)
4. CodeGemma 7B (Google quality)
5. Phi-3.5-mini (surprisingly strong for its size)

### Reasoning / Math / Logic
1. DeepSeek R1 Distill 32B (best accessible reasoning)
2. QwQ-32B (Qwen's reasoning model)
3. DeepSeek R1 Distill 7B (if RAM-constrained)
4. Phi-4 14B (strong reasoning for size)

### RAG (Retrieval Augmented Generation)
1. Command-R 7B (purpose-built for RAG with citations)
2. LLaMA 3.1 8B (strong context understanding)
3. Mistral 7B (used in your project — good choice)
4. Qwen 2.5 7B (128K context — can fit entire documents)

### Document Extraction / Structured Output
1. Mistral 7B Instruct (instruction following, used in your project)
2. LLaMA 3 8B Instruct
3. Qwen 2.5 7B (JSON mode support)

### Multilingual (Indian Languages — Hindi, Telugu, Tamil, etc.)
1. Qwen 2.5 7B/72B (strongest multilingual)
2. LLaMA 3.1 8B (covers many Indian scripts)
3. Aya 8B (Cohere, 101 languages)
4. Gemma 2 9B (Google quality, decent multilingual)

### Long Documents (>10K tokens)
1. Qwen 2.5 7B (128K context)
2. LLaMA 3.1 8B (128K context)
3. Command-R 35B (128K, RAG-tuned)
4. Mistral NeMo 12B (128K)

### Vision / Multimodal
1. LLaMA 3.2 11B Vision (Meta's open multimodal)
2. Qwen-VL 7B (strong vision-language)
3. PaliGemma 3B (Google, efficient)
4. Gemma 2 9B VLM variant

### Edge / Embedded (very limited resources)
1. Phi-3-mini 3.8B (best quality at 3.8B)
2. Qwen 2.5 1.5B (compact, multilingual)
3. LLaMA 3.2 1B/3B (Meta's tiny models)
4. Gemma 2 2B (Google quality, 2B)

---

## 17. Quantization — Making Models Smaller

Quantization reduces the numerical precision of model weights to make models smaller and faster.

### Precision levels
| Format | Bits per weight | Memory (7B model) | Quality loss |
|---|---|---|---|
| FP32 | 32-bit float | ~28 GB | None (full precision) |
| FP16 / BF16 | 16-bit float | ~14 GB | Negligible |
| Q8_0 | 8-bit int | ~7 GB | Very small (<1%) |
| Q5_K_M | 5-bit int | ~5 GB | Small (~1-2%) |
| Q4_K_M | 4-bit int | ~4 GB | Small (~2-3%) — best tradeoff |
| Q3_K_M | 3-bit int | ~3 GB | Moderate (~5%) |
| Q2_K | 2-bit int | ~2.5 GB | Large (10%+) |

### GGUF format (used by Ollama and llama.cpp)
Most quantized models distributed in **GGUF** (GPT-Generated Unified Format). File names like:
```
mistral-7b-instruct.Q4_K_M.gguf
│                    └── Quantization type
│                        K = K-quant (smarter bit allocation)
│                        M = Medium (balance of speed vs quality)
└── Model name
```

### Recommendation for most users
**Q4_K_M** — the standard choice. 4-bit with smarter bit allocation for important layers. Roughly 90% of the full-precision model quality at 25% of the memory.

### How Ollama handles this
When you run `ollama pull mistral`, Ollama automatically downloads the Q4_K_M quantized version and serves it via llama.cpp. You don't need to think about this — it just works.

---

## 18. Running Models Locally — Tools

### Ollama (what your project uses)
```bash
# Install (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Pull and run models
ollama pull mistral        # 4.1GB Q4
ollama pull llama3.1       # 4.7GB Q4
ollama pull qwen2.5:7b     # 4.7GB Q4
ollama pull phi3:mini      # 2.3GB Q4
ollama pull deepseek-r1:7b # 4.7GB Q4

# Run interactively
ollama run mistral

# API (same as your rag_engine.py uses)
# POST http://localhost:11434/api/generate
```

Ollama gives you an OpenAI-compatible API endpoint — drop-in replacement.

### LM Studio
- GUI application for Windows/Mac/Linux
- Point-and-click model download from Hugging Face
- Built-in chat interface
- Serves a local OpenAI-compatible API
- Best for non-technical users

### llama.cpp
- The underlying C++ engine that Ollama and LM Studio use
- Run directly from command line
- Most control over quantization and parameters
- Good for custom deployment

### vLLM (for production/GPU)
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct
```
- Production-grade GPU inference server
- PagedAttention for efficient KV cache management
- Handles multiple concurrent requests
- OpenAI-compatible API

### Hugging Face Transformers (Python)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
```
Best for research and fine-tuning.

---

## 19. Model Naming Conventions Explained

Understanding model names helps you pick the right variant:

```
llama3.1-8b-instruct-q4_k_m.gguf
│         │  │         └── Quantization (see section 17)
│         │  └── "instruct" = instruction-tuned (follows commands)
│         │      "base" = raw pretrained (just predicts next token)
│         │      "chat" = tuned for conversation
│         │      "coder" = tuned for code
│         └── 8 billion parameters
└── Model family and version

mistral-7b-v0.3
│         └── Version 0.3 (improvements over original)
└── 7 billion params

qwen2.5-72b-instruct-awq
│               └── AWQ quantization (alternative to GGUF/Q4)
└── 72 billion params

deepseek-r1-distill-qwen-7b
│          │       └── Built on Qwen 7B as base
│          └── Distilled from R1 (smaller version of R1's reasoning)
└── R1 model family
```

### "Instruct" vs "Base"
- **Base model:** Trained only on predicting next token. Won't follow instructions — it will try to complete your prompt as if it's more text.
- **Instruct/Chat model:** Fine-tuned on (instruction, response) pairs. Understands to "answer the question" not "continue the text."

**Always use Instruct variants for applications.** Base models are for researchers and fine-tuners.

---

## 20. Quick Selection Guide — Decision Tree

```
What do you need?
│
├── Run on phone/edge device
│   └── Phi-3-mini 3.8B or LLaMA 3.2 3B
│
├── Run on laptop (CPU only)
│   ├── 8GB RAM → Phi-3-mini Q4 (3–5 tok/s)
│   ├── 16GB RAM → Mistral 7B Q4 (4–7 tok/s) ← best balance
│   └── 32GB RAM → Phi-3-medium 14B or Qwen 2.5 14B
│
├── Run on GPU
│   ├── 8GB VRAM → Mistral 7B or LLaMA 3 8B (full Q4, fast)
│   ├── 16GB VRAM → LLaMA 3 8B FP16 or 13B Q4
│   ├── 24GB VRAM → Qwen 2.5 32B or Mixtral 8×7B Q4
│   └── 40GB+ VRAM → LLaMA 3 70B or Qwen 2.5 72B
│
├── Best quality (cost no object, cloud/server)
│   ├── Reasoning → DeepSeek R1 or QwQ-32B
│   ├── General → LLaMA 3.1 405B or Qwen 2.5 72B
│   └── Coding → DeepSeek Coder V2 or Qwen 2.5-Coder 32B
│
├── Specific task
│   ├── RAG/Document Q&A → Command-R7B or Mistral 7B (your project)
│   ├── Code generation → Qwen 2.5-Coder or DeepSeek Coder
│   ├── Math/Reasoning → DeepSeek R1 Distill or QwQ
│   ├── Multilingual → Qwen 2.5 or LLaMA 3.1
│   └── Vision → LLaMA 3.2 Vision or Qwen-VL
│
└── Commercial license concern
    ├── Must be Apache 2.0 → Mistral 7B, Falcon 7B, Gemma 2
    ├── MIT License → Phi series, DeepSeek models
    └── Llama License (mostly open) → LLaMA 3 family
```

---

## 21. Common Interview Questions & Answers

**Q: What is the difference between a base model and an instruct model?**  
A: A base model is trained purely on next-token prediction — it learns to continue text statistically. Given a question, it tries to write more text that looks like the training data, not actually answer it. An instruct model is fine-tuned on (instruction, response) pairs using SFT and/or RLHF — it learns to understand "you are an assistant, answer this question." For any application, you always use the instruct variant.

**Q: Why can a 7B model sometimes outperform a 13B model?**  
A: Parameter count is only one factor. Architecture efficiency (GQA, SWA), quality and quantity of training data, and quality of instruction tuning all matter enormously. Mistral 7B beats LLaMA 2 13B because of architectural innovations and better training. Gemma 2 9B beats LLaMA 2 70B in many benchmarks because Google uses extremely high-quality training data. "Bigger isn't always better" is a core lesson of modern LLMs.

**Q: What is quantization and why does it matter?**  
A: Quantization reduces the numerical precision of model weights — for example, from 32-bit floats to 4-bit integers. A Mistral 7B model needs 14GB in FP16 but only ~4GB in Q4. The quality loss at Q4 is only 2–3%, but the memory reduction is 3.5×. This makes running models on consumer hardware practical. Without quantization, most LLMs would require expensive enterprise GPUs.

**Q: What is Mixture of Experts (MoE) and which models use it?**  
A: MoE replaces the single FFN in each Transformer block with multiple "expert" FFNs. A router selects the top 2 experts for each token. This gives a large total parameter count (quality) at the compute cost of a much smaller model. Mixtral 8×7B has 46.7B total params but only activates ~13B per token. DeepSeek V3/R1 have 671B total but only 37B active. The tradeoff: you need enough memory to hold ALL weights, even inactive ones.

**Q: What is the KV cache and why does it matter for deployment?**  
A: The KV cache stores Key and Value tensors computed for all previously processed tokens. Without it, generating each new token would require reprocessing the entire sequence — O(n²) cost. With the cache, each new token only needs to compute its own K,V and attend to the cached previous ones — O(n) cost. The cache size grows with sequence length and limits how long a conversation can be before running out of memory. GQA and MLA reduce KV cache size significantly.

**Q: Which model would you use for a financial document chatbot running on CPU?**  
A: Mistral 7B Instruct, quantized to Q4_K_M, served via Ollama. It runs in ~4GB RAM, generates ~5 tokens/second on a modern laptop CPU (usable for development), has Apache 2.0 license (no commercial restrictions), follows instructions well enough to answer "only from the document" directives, and has strong English extraction capability. This is exactly what the project's `rag_engine.py` uses. If performance is critical, upgrade to a GPU (RTX 3060 = ~40 tok/s) or switch to Phi-3-mini for faster CPU inference.

**Q: What is RAG and why do you need a model for it?**  
A: RAG = Retrieval Augmented Generation. Instead of relying on a model's trained knowledge, you retrieve relevant text chunks from a document, inject them into the prompt, and ask the model to answer using only that context. The LLM's role in RAG is: read the provided context, understand the question, extract the answer. For this, you need a model with strong instruction-following and reading comprehension — not the largest or most "knowledgeable" model. Mistral 7B and Command-R are ideal because they follow "answer only from context" instructions reliably.

**Q: What's special about DeepSeek R1?**  
A: DeepSeek R1 is significant for two reasons: performance and training method. On reasoning benchmarks, it matches OpenAI's o1 — the best reasoning model from a well-funded US company — despite DeepSeek being a smaller Chinese startup. The training method used pure reinforcement learning (no supervised fine-tuning) — the model developed chain-of-thought reasoning spontaneously from RL on math and code problems. It was released open-source with MIT license, causing significant industry discussion about the cost efficiency of frontier model training.

**Q: What does "context length" mean practically?**  
A: Context length is the maximum number of tokens the model can process at once — both input and output combined. 1 token ≈ 4 characters ≈ 0.75 words. Practical equivalents: 4K tokens ≈ 6 pages, 8K ≈ 12 pages, 128K ≈ 200 pages (a full book). For invoice extraction, most invoices fit in 1–2K tokens. For bank statement analysis (transaction history), you might need 8K–32K. For analyzing an entire contract or financial report, 128K context (LLaMA 3.1, Qwen 2.5) becomes necessary.

**Q: Why is Apple Silicon (M-series) good for running LLMs locally?**  
A: Apple Silicon uses unified memory architecture — the CPU and GPU share the same memory pool. An M2 Max with 96GB unified memory gives the GPU access to all 96GB, whereas a PC with 64GB RAM + 12GB VRAM GPU gives the GPU only 12GB. This means a Mac with 32–96GB RAM can run much larger models on the GPU than a PC with a comparable discrete GPU. The M3 Max can run LLaMA 3 70B quantized at 20–30 tok/s — competitive with a mid-range data center GPU.

---

*End of LLM Models Complete Reference Guide*  
*Covers: LLaMA, Mistral, Mixtral, Qwen, Gemma, Phi, Falcon, DeepSeek, Command-R*
