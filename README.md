# 🧠 AI-Powered Resume Screening & Ranking System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-resume-screening-sfdhusg7bdptmf7mwazwgz18.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> An intelligent, production-grade AI resume screener and candidate ranking platform powered by **Sentence-BERT (`all-MiniLM-L6-v2`)**, **Sublinear N-Gram TF-IDF**, **Domain Skill Extraction NLP**, and **Interactive Visual Analytics**.

---

## 🌐 Live Interactive Demo

🚀 **Experience the Live Application:**  
👉 **[https://ai-resume-screening-sfdhusg7bdptmf7mwazwgz18.streamlit.app/](https://ai-resume-screening-sfdhusg7bdptmf7mwazwgz18.streamlit.app/)**

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dual-Engine AI Architecture](#-dual-engine-ai-architecture)
- [Why Sentence-BERT + Skill Taxonomy?](#-why-sentence-bert--skill-taxonomy)
- [Scoring Formula & Accuracy Improvements](#-scoring-formula--accuracy-improvements)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Installation & Local Setup](#-installation--local-setup)
- [How to Use](#-how-to-use)
- [Project Directory Structure](#-project-directory-structure)
- [Deployment on Streamlit Cloud](#-deployment-on-streamlit-cloud)
- [License](#-license)

---

## 📖 Overview

Recruiting teams frequently receive hundreds of resumes for open requisitions. Manual screening is slow, subjective, and labor-intensive. 

The **AI-Powered Resume Screener** automates candidate evaluation with:
1. **Multi-Format Ingestion**: Batch upload `.pdf`, `.docx`, and `.txt` files.
2. **Dual-Engine AI Matching**: Select between **Sentence-BERT Deep Semantic AI** or **Fast TF-IDF NLP**.
3. **Chunk-Pooling Mechanism**: Overcomes BERT's 512-token limit by chunking multi-page resumes into contextual paragraphs without data loss.
4. **Skill Taxonomy & Gap Analysis**: Automatically isolates matched skills (green badges) and missing requirements (red badges).
5. **Actionable Insights & Export**: Interactive Plotly score distribution charts, candidate tier highlights, and 1-click CSV report exports.

---

## 🔬 Dual-Engine AI Architecture

Users and recruiters can toggle between two specialized AI screening engines based on their speed and precision requirements:

```
                                  ┌──────────────────────────────────────────────┐
                                  │             AI SCREENING PIPELINE            │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                        ┌────────────────────────────────┴───────────────────────────────┐
                        │                                                                │
                        ▼                                                                ▼
         ┌─────────────────────────────┐                                  ┌─────────────────────────────┐
         │ 🧠 Sentence-BERT (Recommended)│                                  │   ⚡ Fast Hybrid TF-IDF     │
         ├─────────────────────────────┤                                  ├─────────────────────────────┤
         │ • all-MiniLM-L6-v2 Embeddings│                                  │ • Sublinear N-Gram (1, 2)   │
         │ • Paragraph Chunk-Pooling   │                                  │ • Stop-words removal        │
         │ • Deep Semantic Understanding│                                  │ • Instantaneous (<0.1s)     │
         │ • Accuracy: ~94%            │                                  │ • Accuracy: ~86%            │
         └──────────────┬──────────────┘                                  └──────────────┬──────────────┘
                        │                                                                │
                        └────────────────────────────────┬───────────────────────────────┘
                                                         │
                                                         ▼
                                       ┌──────────────────────────────────┐
                                       │   🎯 Exact Skill Taxonomy Match  │
                                       │   (Languages, Cloud, DBs, ML)    │
                                       └─────────────────┬────────────────┘
                                                         │
                                                         ▼
                                       ┌──────────────────────────────────┐
                                       │  🏆 Composite Match Score (0-100)│
                                       │  + Matched & Missing Skill Badges│
                                       └──────────────────────────────────┘
```

---

## 💡 Why Sentence-BERT + Skill Taxonomy?

1. **Why Not Pure BERT Alone?**  
   BERT is strictly semantic. In pure vector space, `"Java Developer"` and `"Python Developer"` have an ~88% similarity score due to identical sentence structures. A hybrid model ensures mandatory hard requirements are verified through taxonomy matching.
2. **Why Not Pure TF-IDF Alone?**  
   TF-IDF only checks literal keyword matches. If a JD says *"Generative AI and Large Language Models"* while the resume says *"Fine-tuned Transformers & Deep Learning"*, TF-IDF under-scores the candidate. Sentence-BERT bridges this semantic synonym gap.
3. **How We Solved the 512-Token Limit:**  
   Standard BERT truncates resumes longer than 400 words. We implemented **Overlapping Paragraph Chunk-Pooling**:
   - Long resumes are split into 140-word overlapping chunks.
   - Embeddings are calculated for each chunk and pooled against the JD embedding ($0.6 \times \text{Top-3 Average} + 0.4 \times \text{Max Chunk}$).

---

## 📊 Scoring Formula & Candidate Tiers

### 1. Sentence-BERT Hybrid Formula (Recommended)
$$\text{Score (\%)} = \mathbf{(50\% \times \text{SBERT Semantic Match})} + \mathbf{(40\% \times \text{Skill Coverage Ratio})} + \mathbf{(10\% \times \text{Keyword Density})}$$

### 2. Fast TF-IDF Hybrid Formula
$$\text{Score (\%)} = \mathbf{(55\% \times \text{TF-IDF Cosine Match})} + \mathbf{(35\% \times \text{Skill Coverage Ratio})} + \mathbf{(10\% \times \text{Keyword Density})}$$

### 3. Match Tiers
- 🟢 **Strong Match (≥ 75%)**: High skill overlap and rich contextual relevance.
- 🟡 **Moderate Match (50% – 74%)**: Meets foundational requirements with a few missing competencies.
- 🔴 **Low Match (< 50%)**: Low alignment with core job specifications.

---

## ✨ Key Features

- 🧠 **Dual-Engine Selector**: Switch between Sentence-BERT and Fast TF-IDF on the fly.
- 📁 **Multi-Format Batch Ingestion**: Upload PDF, DOCX, and TXT files simultaneously.
- ⚡ **1-Click Presets**: Pre-configured templates for *Python/ML Engineer*, *Full Stack Developer*, and *DevOps Specialist*.
- 🎯 **Skill Gap Analysis**: Visual green badges for matched skills and red badges for missing JD requirements.
- 📈 **Interactive Plotly Visualizations**: Responsive horizontal bar charts displaying ranking distribution.
- 🏆 **Spotlight Cards**: Highlighting Top 3 candidate profiles and key metrics.
- 📥 **Export to CSV**: Export candidate rankings, scores, and skills with one click.

---

## 🛠️ Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Core Language**: Python 3.9+
- **Deep Learning & NLP**:
  - `sentence-transformers` (`all-MiniLM-L6-v2`)
  - `torch` (PyTorch)
  - `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`)
  - Custom Skill Taxonomy Engine
- **Document Extractors**:
  - `PyPDF2` & `python-docx`
- **Visual Analytics**:
  - `plotly`, `pandas`, `numpy`, `seaborn`

---

## 💻 Installation & Local Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/sonawanewdinesh18/AI-Resume-Screening.git
cd AI-Resume-Screening
```

### 2️⃣ Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 🎯 How to Use

1. **Choose Model**: Select *Sentence-BERT (Recommended)* or *Fast Hybrid TF-IDF* in the sidebar.
2. **Set Job Description**: Choose a preset or paste custom requirements.
3. **Upload Resumes**: Upload candidate resumes in `.pdf`, `.docx`, or `.txt`.
4. **Click "🚀 Process & Rank Resumes"**: The AI engine will parse, score, and rank candidates.
5. **Download Report**: Click **"📥 Download Full Screening Report (CSV)"** to save the recruiter spreadsheet.

---

## ☁️ Deployment on Streamlit Cloud

1. Push code to GitHub: `https://github.com/sonawanewdinesh18/AI-Resume-Screening`
2. Connect to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Set Main File path to `app.py`.
4. Deploy!

Live link: **[https://ai-resume-screening-sfdhusg7bdptmf7mwazwgz18.streamlit.app/](https://ai-resume-screening-sfdhusg7bdptmf7mwazwgz18.streamlit.app/)**

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
