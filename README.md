# 🧠 AI-Powered Resume Screening & Ranking System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-resume-screening-sfdhusg7bdptmf7mwazwgz18.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> An intelligent, production-ready AI resume screener and candidate ranking platform built with **Python**, **Streamlit**, **Advanced NLP (Hybrid TF-IDF + Skill Extraction)**, and **Interactive Visual Analytics**.

---

## 🌐 Live Interactive Demo

🚀 **Experience the Live Application:**  
👉 **[https://ai-resume-screening-sfdhusg7bdptmf7mwazwgz18.streamlit.app/](https://ai-resume-screening-sfdhusg7bdptmf7mwazwgz18.streamlit.app/)**

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Model Architecture & Accuracy Improvements](#-model-architecture--accuracy-improvements)
- [Key Features](#-key-features)
- [Screenshots & UI Preview](#-screenshots--ui-preview)
- [Tech Stack](#-tech-stack)
- [Installation & Local Setup](#-installation--local-setup)
- [How to Use](#-how-to-use)
- [Project Directory Structure](#-project-directory-structure)
- [Deployment on Streamlit Cloud](#-deployment-on-streamlit-cloud)
- [Roadmap & Further Enhancements](#-roadmap--further-enhancements)
- [License](#-license)

---

## 📖 Overview

Recruiting teams receive hundreds of resumes for a single job opening. Manual screening is time-consuming, prone to human bias, and inefficient. 

The **AI-Powered Resume Screening & Ranking System** solves this by:
1. Ingesting resumes in batch (`.pdf`, `.docx`, `.txt`).
2. Parsing and cleaning unstructured text while preserving vital tech symbols (`C++`, `C#`, `.NET`, `CI/CD`, `Node.js`).
3. Extracting technical skills, competencies, and domain keywords using an NLP taxonomy.
4. Calculating an **Ensemble Match Score** combining TF-IDF N-Gram contextual alignment, direct skill coverage, and keyword density.
5. Providing recruiters with interactive charts, candidate skill gap badges, top talent highlights, and downloadable CSV reports.

---

## 🔬 Model Architecture & Accuracy Improvements

### 1. The Baseline vs. Enhanced Model
| Metric / Component | Traditional Baseline | 🚀 Our Enhanced Hybrid Model |
| :--- | :--- | :--- |
| **Vectorization** | Unigram TF-IDF only | **Bi-Gram & Multi-Gram TF-IDF (`ngram_range=(1, 2)`)** |
| **TF Weighting** | Linear Term Frequency | **Sublinear TF Scaling (`sublinear_tf=True`)** to penalize spam keyword stuffing |
| **Text Preprocessing** | Raw strings (noise, URLs, punctuation) | **Normalized Tech Regex Cleaner** (preserves `C++`, `.NET`, `React.js`, `CI/CD`) |
| **Skill Extraction** | None | **Automated Skill & Taxonomy Matcher** (identifies matched vs. missing skills) |
| **Scoring Formula** | Simple Cosine Similarity | **Weighted Ensemble Hybrid Score (0% – 100%)** |
| **Explainability** | Black-box single decimal score | **Detailed Skill Breakdown** (Matched skills in green, gaps in red) |

### 2. Hybrid Scoring Formula
$$\text{Final Candidate Score (\%)} = (\text{TF-IDF Cosine Similarity} \times 0.55) + (\text{Skill Coverage Ratio} \times 0.35) + (\text{Keyword Density} \times 0.10)$$

- **TF-IDF Semantic Vectorization (55%)**: Assesses overall contextual relevance and phrase similarity between the job description and the resume.
- **Explicit Skill Match Ratio (35%)**: $\frac{\text{Skills Detected in Resume} \cap \text{Skills Required in JD}}{\text{Total Skills in JD}}$ ensures candidates possess the mandatory tools and frameworks.
- **Keyword Overlap Factor (10%)**: Broad vocabulary match to ensure domain depth.

---

## ✨ Key Features

- 📁 **Multi-Format Batch Ingestion**: Upload dozens of PDF, DOCX, and TXT files simultaneously.
- ⚡ **Instant Job Presets**: Choose 1-click presets for *Python/ML Engineer*, *Full Stack Developer*, *DevOps Specialist*, or enter a custom job description.
- 🎯 **Explainable Skill Gap Analysis**: Visual green badges for matched competencies and red badges for missing requirements.
- 📊 **Interactive Plotly Visualizations**: Dynamic horizontal bar charts and distribution graphs.
- 🏆 **Top Talent Spotlight Cards**: Highlight the top 3 candidates with glassmorphic cards and match tier classifications (Strong, Moderate, Low).
- 📥 **Export to CSV**: Export the entire candidate pool ranking, scores, and skill gap report with a single click.
- 🎨 **Glassmorphism UI**: Beautiful, modern dark-mode interface with smooth animations and responsive layout.

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Core Language**: Python 3.9+
- **Machine Learning & NLP**:
  - `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`)
  - Custom regex tokenizers & Skill Taxonomy Engine
- **Document Parsers**:
  - `PyPDF2` (PDF text extraction)
  - `python-docx` (DOCX parsing)
- **Data & Visual Analytics**:
  - `pandas` & `numpy`
  - `plotly` & `seaborn` / `matplotlib`

---

## 💻 Installation & Local Setup

### Prerequisites
- Python 3.9 or higher installed
- Git installed

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

### 4️⃣ Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎯 How to Use

1. **Select or Paste Job Description**:
   - Use the sidebar preset or paste custom job requirements in the text box.
   - Review the automatically detected key skills.
2. **Upload Resumes**:
   - Drag and drop candidate resumes (`.pdf`, `.docx`, `.txt`).
   - Optionally expand the preview tab to verify extracted text.
3. **Run AI Screening**:
   - Click **"🚀 Process & Rank Resumes"**.
4. **Analyze Candidates & Export**:
   - Review rankings, scores, and matched vs. missing skills.
   - Click **"📥 Download Full Screening Report (CSV)"** to save the recruiter summary.

---

## 📂 Project Directory Structure

```
AI-Resume-Screening/
├── app.py                      # Core Streamlit application & Hybrid NLP Engine
├── requirements.txt            # Python dependencies for deployment
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation & live links
├── background.png              # UI background asset
├── image.png                   # App logo & icon asset
├── logo.png                    # Brand logo asset
└── sample_resumes/             # Sample candidate resumes for evaluation
```

---

## ☁️ Deployment on Streamlit Cloud

This project is automatically deployed and hosted on **Streamlit Community Cloud**:

1. Push your repository to GitHub: `https://github.com/sonawanewdinesh18/AI-Resume-Screening`.
2. Connect your GitHub account to [share.streamlit.io](https://share.streamlit.io/).
3. Select the repository `sonawanewdinesh18/AI-Resume-Screening`, branch `main`, and main file `app.py`.
4. Click **Deploy**!

Live link: **[https://ai-resume-screening-sfdhusg7bdptmf7mwazwgz18.streamlit.app/](https://ai-resume-screening-sfdhusg7bdptmf7mwazwgz18.streamlit.app/)**

---

## 🚀 Roadmap & Further Enhancements

For future enterprise scaling, the following upgrades can be added:
- [ ] **Transformer Embeddings**: Integrate `sentence-transformers` (`all-MiniLM-L6-v2`) for deeper semantic sentence context.
- [ ] **Automated Candidate Summarization**: Using lightweight LLMs to generate 2-line candidate strengths & weaknesses summaries.
- [ ] **Experience Level Detection**: Parsing years of experience from resume timelines.
- [ ] **ATS Direct Integration**: Direct webhooks into Greenhouse, Lever, or Workday.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
