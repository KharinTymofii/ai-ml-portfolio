# AI & ML Portfolio — Tymofii Kharin

Hi! I'm Tymofii, an AI student at the Kyiv School of Economics.  
I enjoy working with data, building machine learning systems, and combining research-style thinking with practical projects.

This repo is a curated overview of the ML work I've done so far — course projects, research collaborations, Kaggle experiments, and an industry internship. It’s not meant to be a perfect codebase, but rather an honest snapshot of what I’ve actually tried, broken, fixed, and learned.

---

## What I like working on

- Turning **messy real-world data** into something models can learn from  
- **NLP & RAG** – using language models in a grounded, data-aware way  
- **Computer vision** experiments with modern backbones (EfficientNet, etc.)  
- Building small, useful ML tools that could realistically live inside a product 


---

## Selected Projects

### 1. Multimodal RAG System

A personal project to learn how **retrieval-augmented generation** works in practice, and how to mix text + image context.

- Collected and cleaned a small corpus (e.g. newsletter content)  
- Built a **vector store** with:
  - text embeddings (E5 family)
  - image embeddings (CLIP-like models)
- Used a **Gradio** interface to ask questions and get answers grounded in retrieved content  
- Focused on:
  - making the pipeline transparent
  - keeping retrieval interpretable
  - and understanding where RAG actually fails

Code: separate repo — https://github.com/KharinTymofii/Rag-system

---

### 2. Public Procurement Risk (Prozorro)

Applied ML project on Ukrainian public procurement data (Prozorro).  
The goal was to see whether we can flag tenders that are likely to fail, be delayed, or change price.

My role focused primarily on the **risk-generation pipeline**, which meant working with Prozorro’s official risk rule framework and adapting it to a very large real-world dataset:

- processed ~5 GB of raw JSON tender data  
- split data into manageable parts to avoid memory issues  
- cleaned and normalized the dataset (removing damaged or incomplete tenders)  
- ran official Prozorro risk rules using Dockerized [prozorro-risks](https://github.com/ProzorroUKR/prozorro-risks)  
- extracted and aggregated risk flags for further ML modelling  

This work turned out to be a surprisingly engineering-heavy part of the project — especially figuring out which risk rules work on which objects (tenders vs contracts), and how to make them run efficiently on many tens of thousands of records.

   
📄 Report (PDF): `prozorro-risk-ml/Final_Paper_Tenders.pdf`

---

### 3. EuroVis 2025 — Visualization Literacy Study

A short paper accepted to **EuroVis 2025 (Short Papers)**, where we studied how a visualization literacy test behaves when you translate and adapt it to another culture and language.

In the project I:

- worked with response data from participants  
- compared performance across:
  - original English version,
  - direct translation,
  - culturally adapted Ukrainian version  
- helped analyse which items changed difficulty and why

The project sits at the intersection of **data analysis, education, and visualization**, and it made me think a lot about how standardized tests actually work in different contexts.
  
📄 Paper: https://diglib.eg.org/items/8bdfa4af-99f7-4969-b38c-1dd4be045bcd

---

### 4. Kaggle Experiments

These projects are where I explore validation, error analysis, ensembling, thresholding, and other practical ML topics.

#### 🔹 Mechanisms of Action (MoA) — multi-label tabular ML  
- Hybrid GroupKFold + MultilabelStratifiedKFold  
- MLP FeatureGate model, XGBoost  
- Per-class blending, stacking, threshold optimization  
→ **Full report:** `kaggle-projects/moa-multilabel.md`

#### 🔹 Cats & Dogs++ — multi-label CV  
- EfficientNet-V2-S + EfficientNet-B3 ensemble  
- Heavy augmentations for rare classes  
- Per-class α-blending + coordination descent thresholds  
→ **Full report:** `kaggle-projects/cats-and-dogs++.md`


These projects are where I make most of my mistakes and learn the practical details: validation leakage, bad metrics, wrong folds, etc. Then I try not to repeat them in more serious work.

---

### 5. Telecom Internship — BERT for Account Extraction

During my internship at **Vega Telecom**, I worked on a very concrete NLP problem:

> Given noisy, sometimes multilingual service messages, can we reliably extract the customer’s account number?

I:

- built a simple rule-based baseline (regex + pattern checks)  
- prepared training data for a BERT-based model  
- fine-tuned the model to recognise account number spans  
- compared approaches using precision/recall/F1, and did error analysis

This was my first experience with **production-adjacent text data**, with all the quirks and limitations that come with it.

---

### 6. Other Things

- Anime Recommender + chatbot (TF-IDF + cosine + Django backend)  
- Smaller optimization, visualization, and database projects  

#### Ongoing:
- **KSE UA Location Extraction 2025** (LSTM baseline → transformer comparison)  

---

## Skills at a Glance

**Languages:** Python, SQL  
**ML / DL:** scikit-learn, PyTorch, PyTorch Lightning, XGBoost, LightGBM  
**NLP:** tokenization, embeddings, transformers, BERT, basic RAG setups  
**CV:** EfficientNet, augmentations, multi-label classification, TTA  
**Data:** NumPy, Pandas, Matplotlib, Plotly  
**Dev / Infra:** Git, Docker, Linux basics

---


## Contact

If you’d like to talk about junior AI/ML roles, EdTech ideas, or just compare experiments — feel free to reach out.

- 📧 Email: **tkharin@kse.org.ua**  
- GitHub: **@KharinTymofii**
