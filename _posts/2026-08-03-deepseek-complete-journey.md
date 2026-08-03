---
layout: post
title: "DeepSeek: The Complete Journey — From Hedge Fund to Open-Source AI Pioneer"
date: 2026-08-03 02:30:00 +0300
excerpt: "A comprehensive, fact-checked educational guide to DeepSeek — the Chinese AI lab that trained frontier models for $5.6M and gave them away under MIT license. Every model, every innovation, explained."
categories: [ai, machine-learning, open-source, education]
mathjax: false
---

> **Truth Statement:** This article is written for educational purposes. All claims are sourced from Wikipedia, DeepSeek's official technical reports, peer-reviewed papers, and verified news outlets. Where estimates or projections appear, they are clearly labeled as such.

---

## What Is DeepSeek?

**DeepSeek** (full name: Hangzhou DeepSeek Artificial Intelligence Basic Technology Research Co., Ltd.) is a Chinese artificial intelligence company founded in July 2023 by **Liang Wenfeng**. It is owned and funded by **High-Flyer**, a quantitative {% include term.html name="hedge_fund" text="hedge fund" %} that Liang co-founded in 2015. The company is headquartered in Hangzhou, Zhejiang, China, and employs approximately 160 people as of 2025.

DeepSeek develops {% include term.html name="large_language_model" text="large language models" %} (LLMs) — {% include term.html name="neural_network" text="neural networks" %} trained on vast amounts of text to understand and generate human language. Unlike many AI labs backed by venture capital, DeepSeek operates with a research-first mindset. The company has stated it focuses on research and does not have immediate plans for commercialization. This posture allows it to release its models as **open weights** — meaning anyone can download and run them — under permissive licenses like the MIT License.

> **Key Fact:** DeepSeek's V3 model was trained for approximately **$5.6 million** (2.788 million GPU hours on H800 chips), according to the company's technical report. This is significantly less than the estimated $100 million spent by OpenAI on GPT-4.

---

## The Origin Story: From Trading Algorithms to AI Research

### The Hedge Fund Roots (2015–2023)

Before DeepSeek existed, there was **High-Flyer Capital**, a quantitative hedge fund founded by Liang Wenfeng in June 2015. High-Flyer began using GPU-dependent deep learning models for stock trading as early as October 2016. By the end of 2017, most of its trading was AI-driven.

In 2019, High-Flyer built its first computing cluster, **Fire-Flyer 1**, containing 1,100 GPUs. It was retired after 1.5 years. Then came **Fire-Flyer 2** in 2021, with a budget of 1 billion yuan. By 2022, this cluster had 5,000 Nvidia A100 GPUs.

A crucial detail: Liang reportedly acquired **10,000 Nvidia A100 GPUs** before the United States restricted chip sales to China in 2022. This early stockpile would later become the foundation for DeepSeek's {% include term.html name="training" text="training" %} infrastructure.

### The Spin-Off (2023)

On April 14, 2023, High-Flyer announced the launch of an artificial general intelligence (AGI) research lab. Two months later, on **July 17, 2023**, that lab was spun off into an independent company: **DeepSeek**.

Venture capital investors were initially reluctant to fund DeepSeek, believing it unlikely to generate a quick return. High-Flyer remained the principal investor and backer.

---

## The Model Timeline: Every Release, Explained

![DeepSeek Timeline](/assets/images/deepseek_timeline.gif)
*Animated timeline of DeepSeek's model releases from November 2023 to July 2026. Click to view the full animation.*

### Era 1: Foundations (November 2023)

**DeepSeek Coder** (November 2, 2023) was DeepSeek's first public model. It came in sizes from 1.3 billion to 33 billion {% include term.html name="parameters" text="parameters" %}, trained on 1.8 trillion {% include term.html name="token" text="tokens" %} (87% source code). It supported 16,000-token {% include term.html name="context_window" text="context windows" %}. The architecture was based on Llama, a popular open-source model from Meta.

**DeepSeek-LLM** (November 29, 2023) followed shortly after, with 7 billion and 67 billion parameter variants. The 67B base model reportedly outperformed Llama 2 70B in reasoning, coding, math, and Chinese comprehension. Both models were released under the DeepSeek License (open-weights with some usage restrictions).

> **What are parameters?** Think of parameters as the "knobs and dials" inside a neural network. More parameters generally mean more capacity to learn, but also more computational cost. DeepSeek Coder's 33B variant has 33 billion of these knobs.

### Era 2: Innovation Begins (2024)

**DeepSeek-MoE** (January 2024) introduced a critical architectural innovation: **Shared + Routed Experts**.

![Mixture of Experts Diagram](/assets/images/diagram_moe.png)
*How {% include term.html name="mixture_of_experts" text="Mixture of Experts" %} works: Instead of using all parameters for every token, only relevant "experts" are activated.*

**What is MoE?** A standard neural network uses all its parameters for every input. A {% include term.html name="mixture_of_experts" text="Mixture of Experts" %} (MoE) model divides its parameters into many smaller "expert" networks. A "router" decides which experts to use for each input. DeepSeek's innovation was adding **shared experts** (always active, handling common tasks) alongside **routed experts** (activated on demand).

This matters because:
- DeepSeek-MoE has **16 billion total parameters** but only activates **2.7 billion per token**
- This makes {% include term.html name="inference" text="inference" %} much cheaper without sacrificing capability

**DeepSeek-Math** (April 2024) introduced **GRPO** (Group Relative Policy Optimization), a {% include term.html name="reinforcement_learning" text="reinforcement learning" %} algorithm that does not need a separate "critic" model. It also used a **Process Reward Model (PRM)**, which rewards each reasoning step rather than just the final answer.

**DeepSeek-V2** (May 2024) was a leap forward with **236 billion parameters** (21 billion active). It introduced two major innovations:

1. **Multi-Head Latent Attention (MLA)**: Replaced standard {% include term.html name="attention_mechanism" text="attention" %} with compressed latent vectors, dramatically reducing memory usage during inference.
2. **YaRN**: Extended context length from 4,000 to 128,000 tokens.

![MLA Diagram](/assets/images/diagram_mla.png)
*Multi-Head Latent Attention compresses the massive {% include term.html name="kv_cache" text="KV cache" %} into tiny latent vectors, making inference faster and cheaper.*

**What is the KV cache?** When a language model processes text, it stores "keys" and "values" (mathematical representations) for every token. For long conversations, this cache grows enormous. MLA compresses these into much smaller "latent vectors" — like keeping a compact index card instead of a full library book.

The price was also groundbreaking: **2 RMB per million output tokens**, making it the cheapest high-performance model on the market at the time.

### Era 3: Scale and the R1 Moment (Late 2024–2025)

**DeepSeek-V3** (December 2024) became the model that put DeepSeek on the global map. With **671 billion parameters** (37 billion active), it matched or exceeded GPT-4o and Claude 3.5 Sonnet on many {% include term.html name="benchmark" text="benchmarks" %}.

Key technical achievements in V3:
- **Multi-Token Prediction (MTP)**: Predicts multiple future tokens simultaneously, not just one
- **FP8 Mixed Precision**: Custom 8-bit floating-point format for training, requiring specially written mathematical kernels
- **3FS (Fire-Flyer File System)**: A distributed file system designed for training workloads where data is never read twice
- **hfreduce**: An asynchronous communication library that runs on CPU to avoid blocking GPU work
- **Expert models**: Domain-specific teacher models that generated high-quality synthetic training data

The total training cost was **$5.576 million** — pre-training ($5.328M), context extension ($0.24M), and fine-tuning ($0.01M).

> **Important Caveat:** This $5.6M figure covers only the final training run. It does not include costs for research, failed experiments, infrastructure, salaries, or the earlier models (V2, MoE, etc.) that V3 was built upon. Some analysts have called this figure misleading for this reason.

**DeepSeek-R1** (January 20, 2025) was the "Sputnik moment." It was a dedicated **{% include term.html name="reasoning_model" text="reasoning model" %}** — meaning it generates step-by-step chains of thought before answering, making it especially good at math, coding, and logic puzzles.

R1's training pipeline was:
1. Start with V3-Base
2. "Cold-start" with curated reasoning examples
3. Train R1-Zero using pure reinforcement learning (no human examples)
4. {% include term.html name="distillation" text="Distill" %} R1-Zero's reasoning into smaller models (1.5B to 70B parameters)
5. Final RL with rule-based and preference rewards

R1 was released under the **MIT License**, making it free for commercial use. It surpassed ChatGPT as the #1 free app on the US iOS App Store by January 27, 2025. Nvidia's stock dropped 18% in response, losing approximately **$600 billion in market value** — the largest single-day decline for any company in US stock market history.

> **What is a reasoning model?** Standard AI gives you an answer directly. A reasoning model shows its work — like writing out math steps on paper. This makes it more accurate for complex problems but slower and more expensive for simple ones.

**DeepSeek-V3-0324** (March 2025) and **DeepSeek-R1-0528** (May 2025) were incremental updates improving reasoning, coding, and reducing hallucinations.

### Era 4: The Hybrid Era (2025)

**DeepSeek-V3.1** (August 2025) introduced a **hybrid architecture** — one model that could switch between "thinking" mode (step-by-step reasoning like R1) and "non-thinking" mode (fast direct answers like V3). This was achieved by training on 800 billion additional tokens.

**DeepSeek-V3.2** (December 2025) added **DeepSeek Sparse Attention (DSA)**, a more efficient attention mechanism that selectively attends to past tokens rather than all of them.

### Era 5: The Future Architecture (2026)

**DeepSeek-V4** (April 2026) previewed two models: a **1.6 trillion parameter Pro** and a **284 billion parameter Flash**. Both support a **1 million token context window** — roughly 750,000 words, or the length of several novels.

Key innovations:
- **Compressed Sparse Attention + Hybrid Chunked Attention**
- **Muon optimizer** (a new optimization algorithm)
- Adopted by Huawei and Cambricon for their AI chips

**DSpark** (July 2026) is a {% include term.html name="speculative_decoding" text="speculative decoding" %} technique that makes inference 60–85% faster without retraining or changing outputs.

**V4 Flash 0731** (July 31, 2026) is the current state-of-the-art, scoring **82.7% on Terminal-Bench 2.1** — beating even the 1.6T Pro model on agent tasks. It costs **$0.14 per million input tokens** and **$0.28 per million output tokens**.

---

## The 12 Technical Innovations That Matter

| # | Innovation | When | What It Does |
|---|-----------|------|-------------|
| 1 | **Shared + Routed MoE** | Jan 2024 | Core experts always available; peripheral ones on-demand. Prevents wasted parameters. |
| 2 | **GRPO** | Apr 2024 | Reinforcement learning without a separate critic model. Simpler and more efficient. |
| 3 | **MLA** | May 2024 | Compresses the KV cache into latent vectors. Slashes inference memory and cost. |
| 4 | **MTP** | Dec 2024 | Predicts multiple future tokens at once. Faster decoding, better learning. |
| 5 | **FP8 Mixed Precision** | Dec 2024 | Custom 8-bit number formats. Required writing their own GPU kernels. |
| 6 | **3FS + hfreduce** | Dec 2024 | Custom file system and communication library. Optimized for their exact hardware. |
| 7 | **Two-Stage RL** | May 2024 | First train reasoning with compiler feedback, then general alignment with human preferences. |
| 8 | **Expert Models** | Dec 2024 | Domain-specific teachers generate perfect synthetic training data for math, code, and logic. |
| 9 | **Hybrid Modes** | Aug 2025 | One model handles both fast chat and deep reasoning. No need to choose between models. |
| 10 | **DSA** | Dec 2025 | Sparse attention that selectively focuses on relevant past tokens. Scales better than full attention. |
| 11 | **mHC** | Jan 2026 | Manifold-Constrained Hyper-Connections. A new primitive for scaling without brute-force GPU usage. |
| 12 | **DSpark** | Jul 2026 | Speculative decoding. 60–85% faster inference with identical output quality. |

---

## Global Impact: Why This Matters Beyond Benchmarks

![Impact Diagram](/assets/images/diagram_impact.png)
*DeepSeek's impact spans cost reduction, open access, and global adoption — particularly in developing regions.*

### For the AI Industry

DeepSeek proved that **efficient engineering can match brute-force scaling**. When a 160-person team trains frontier AI for ~$6 million and gives it away for free, the business model of closed-source AI — charging $20/month for {% include term.html name="api" text="API" %} access — comes under pressure.

Competitors responded:
- OpenAI slashed prices following V4 Flash's release
- Claude Sonnet 5 raised prices 50% (August 2025), suggesting the closed-source moat is narrowing

### For China

China's open-source LLMs now account for approximately **one-third of global usage**, up from near-zero in late 2024. As Nvidia CEO Jensen Huang noted, chip restrictions did not slow China — they gave local talent "the spirit, the energy, and the government support to accelerate."

### For Africa and Developing Regions

DeepSeek's usage in Africa is reported to be **2–4x higher** than in other regions. The reasons are practical:

- **Cost**: At $0.14 per million tokens, it is accessible to startups and students
- **No vendor lock-in**: Open weights mean local deployment
- **Data sovereignty**: Models can run on local servers (e.g., Huawei Cloud in Africa)
- **Language support**: DeepSeek has bolstered African language models

Startups in Nairobi and other cities are already building on DeepSeek weights. With local data centers growing (Kenya has 18, Nigeria 16, South Africa leads with 49), the infrastructure is catching up.

> **Note on Privacy Concerns:** Some experts have raised alarms about data privacy when using DeepSeek through Chinese cloud providers. User prompts and locations may be stored on servers accessible by the Chinese government. For child-centered or sensitive applications, local deployment of open-weight models on edge devices is recommended.

---

## The Full Infographic

![DeepSeek Complete Infographic](/assets/images/deepseek_infographic.png)
*A static infographic summarizing DeepSeek's journey, innovations, and impact. Right-click to open in full size.*

---

## Frequently Asked Questions

**Q: Is DeepSeek really "open source"?**

A: DeepSeek releases **open weights** — the trained model parameters are downloadable. However, the training data and full training code are not always public. Since 2025, models have been released under the MIT License, which is a true {% include term.html name="open_source" text="open-source" %} software license. This is more open than "open weights" alone, but less open than releasing the entire training pipeline.

**Q: Can DeepSeek models run on my laptop?**

A: The distilled versions (1.5B to 70B parameters) can run on consumer hardware. The full 671B parameter models require data-center GPUs. The 1.5B distilled model can run on a modern smartphone.

**Q: Is DeepSeek safe to use?**

A: Like all AI models, DeepSeek has limitations. The web interface includes content moderation to comply with Chinese regulations. The downloadable models have no such restrictions. As with any AI, outputs should be verified, especially for factual or medical queries.

**Q: How does DeepSeek compare to GPT-4 or Claude?**

A: On many benchmarks (math, coding, reasoning), DeepSeek-R1 and V3 match or exceed GPT-4o and Claude 3.5 Sonnet. On general knowledge and creative writing, results vary by task. The key difference is cost: DeepSeek is roughly 10–30x cheaper to run.

**Q: What hardware does DeepSeek use?**

A: DeepSeek trains on Nvidia H800 GPUs (a restricted-export version of the H100). They have also begun working with Huawei and Cambricon chips for inference. The company does not use the most advanced chips available to US labs.

---

## Sources and Further Reading

This article is based on the following verified sources:

1. **Wikipedia — DeepSeek**: [https://en.wikipedia.org/wiki/DeepSeek](https://en.wikipedia.org/wiki/DeepSeek) — Company history, model specifications, and timeline.
2. **DeepSeek-V3 Technical Report** (arXiv): Training costs, architecture details, and benchmark results.
3. **DeepSeek API Documentation**: Official changelog confirming release dates and feature updates.
4. **Timeline of DeepSeek** (timelines.issarice.com): Detailed chronological breakdown of events.
5. **BentoML Guide to DeepSeek Models**: Technical comparison of V3, R1, and subsequent versions.
6. **Sebastian Raschka's Technical Tour**: In-depth analysis of MLA, DSA, and training methodologies.
7. **ADF Magazine**: Coverage of DeepSeek's expansion in Africa and associated privacy concerns.
8. **Financial Times / Bloomberg**: Reporting on IPO preparations (July 2026) and funding rounds.

> **Correction Policy:** If you find any factual errors in this article, please open an issue on this repository. Truth matters more than speed.

---

*Last updated: August 3, 2026. This article is published under the same spirit as DeepSeek itself: open, educational, and accessible to all.*
