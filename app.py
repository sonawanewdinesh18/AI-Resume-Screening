import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import re
from PyPDF2 import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screener & Ranker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Common Tech Skills & Domain Keywords Taxonomy
# ---------------------------------------------------------
SKILL_TAXONOMY = [
    # Programming Languages
    "python", "java", "c++", "c#", "c", "javascript", "typescript", "ruby", "php", "swift",
    "kotlin", "go", "golang", "rust", "scala", "r", "dart", "matlab", "bash", "shell",
    # Web Frameworks & Libraries
    "react", "react.js", "angular", "vue", "vue.js", "next.js", "nuxt.js", "node.js", "express",
    "django", "flask", "fastapi", "spring", "spring boot", "asp.net", ".net core", ".net",
    "laravel", "ruby on rails", "html", "html5", "css", "css3", "sass", "bootstrap", "tailwind",
    # AI / ML / Data Science
    "machine learning", "deep learning", "nlp", "natural language processing", "computer vision",
    "artificial intelligence", "generative ai", "llm", "large language models", "tensorflow",
    "pytorch", "keras", "scikit-learn", "sklearn", "pandas", "numpy", "scipy", "opencv",
    "spacy", "nltk", "hugging face", "transformers", "langchain", "llamaindex", "bert", "gpt",
    "data science", "data analysis", "data mining", "feature engineering", "data visualization",
    "matplotlib", "seaborn", "plotly", "power bi", "tableau", "excel", "bigquery", "spark", "hadoop",
    # Cloud & DevOps
    "aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud platform",
    "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins", "gitlab ci", "github actions",
    "ci/cd", "continuous integration", "continuous deployment", "linux", "ubuntu", "nginx", "apache",
    "serverless", "microservices", "helm", "prometheus", "grafana",
    # Databases & Caching
    "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle", "mongodb", "nosql", "redis",
    "cassandra", "elasticsearch", "dynamodb", "firebase", "supabase", "mariadb", "neo4j",
    # Software Engineering & Methodologies
    "git", "github", "gitlab", "bitbucket", "rest api", "graphql", "grpc", "soap",
    "system design", "distributed systems", "oop", "object oriented programming",
    "data structures", "algorithms", "agile", "scrum", "kanban", "jira", "unit testing",
    "pytest", "selenium", "cypress", "tdd", "test driven development", "clean code",
    # Cybersecurity & Networking
    "cybersecurity", "penetration testing", "cryptography", "oauth", "jwt", "ssl", "tls",
    "firewall", "siem", "soc", "network security", "vulnerability assessment"
]

# ---------------------------------------------------------
# Helper Functions: File Reading & Text Extraction
# ---------------------------------------------------------
def extract_text_from_pdf(pdf_file) -> str:
    """Extracts clean text safely from an uploaded PDF file."""
    text = ""
    try:
        pdf_file.seek(0)
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.warning(f"Warning extracting text from PDF ({getattr(pdf_file, 'name', 'unknown')}): {e}")
    return text.strip()


def extract_text_from_docx(docx_file) -> str:
    """Extracts clean text safely from an uploaded DOCX file."""
    text = ""
    try:
        docx_file.seek(0)
        doc = docx.Document(docx_file)
        text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except Exception as e:
        st.warning(f"Warning extracting text from DOCX ({getattr(docx_file, 'name', 'unknown')}): {e}")
    return text.strip()


def extract_text_from_txt(txt_file) -> str:
    """Extracts clean text safely from an uploaded TXT file."""
    text = ""
    try:
        txt_file.seek(0)
        raw = txt_file.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="ignore")
    except Exception as e:
        st.warning(f"Warning reading TXT file ({getattr(txt_file, 'name', 'unknown')}): {e}")
    return text.strip()


def extract_resume_text(uploaded_file) -> str:
    """Dispatches appropriate parser based on file mime or extension."""
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".pdf") or uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_name.endswith(".docx") or "officedocument.wordprocessingml" in uploaded_file.type:
        return extract_text_from_docx(uploaded_file)
    else:
        return extract_text_from_txt(uploaded_file)


# ---------------------------------------------------------
# NLP Preprocessing & Skill Extraction Engine
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """Cleans text while preserving essential technical characters (+, #, ., /)."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'c\+\+', 'cpp', text)
    text = re.sub(r'c\#', 'csharp', text)
    text = re.sub(r'\.net', 'dotnet', text)
    text = re.sub(r'node\.js', 'nodejs', text)
    text = re.sub(r'react\.js', 'reactjs', text)
    text = re.sub(r'vue\.js', 'vuejs', text)
    text = re.sub(r'ci\/cd', 'cicd', text)
    
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_skills(text: str) -> set:
    """Extracts known skills and domain keywords from text using boundary matching."""
    if not text:
        return set()
    
    lower_text = " " + text.lower() + " "
    found_skills = set()
    
    for skill in SKILL_TAXONOMY:
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(skill) + r'(?![a-zA-Z0-9])'
        if re.search(pattern, lower_text):
            found_skills.add(skill)
            
    return found_skills


# ---------------------------------------------------------
# Sentence-BERT Model Loader & Chunk Pooling
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sbert_model():
    """Lazily loads and caches the Sentence-BERT all-MiniLM-L6-v2 model."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model, True
    except Exception:
        return None, False


def chunk_text(text: str, max_words: int = 140) -> list:
    """Chunks long text into overlapping paragraph blocks to avoid token limits."""
    words = text.split()
    if not words:
        return [""]
    if len(words) <= max_words:
        return [text]
    
    step = max(40, int(max_words * 0.7))
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks


def compute_sbert_similarity(model, job_description: str, resume_texts: list) -> list:
    """Computes semantic similarity with chunk pooling for multi-page resumes."""
    from sentence_transformers import util
    
    jd_embedding = model.encode(job_description, convert_to_tensor=True, normalize_embeddings=True)
    scores = []
    
    for text in resume_texts:
        chunks = chunk_text(text, max_words=140)
        chunk_embeddings = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
        chunk_sims = util.cos_sim(jd_embedding, chunk_embeddings)[0].cpu().numpy()
        
        # Max chunk similarity + Top-3 average chunk similarity
        top_k = min(3, len(chunk_sims))
        sorted_sims = np.sort(chunk_sims)
        top_k_avg = float(np.mean(sorted_sims[-top_k:]))
        max_sim = float(sorted_sims[-1])
        
        combined_sim = max(0.0, (top_k_avg * 0.6) + (max_sim * 0.4))
        scores.append(combined_sim)
        
    return scores


# ---------------------------------------------------------
# Dual-Engine AI Scoring Pipeline
# ---------------------------------------------------------
def calculate_scores(job_description: str, resume_data: list, use_sbert: bool = True) -> tuple:
    """
    Computes ensemble matching score using Sentence-BERT (or TF-IDF fallback) + Skill Taxonomy.
    """
    jd_skills = extract_skills(job_description)
    cleaned_jd = clean_text(job_description)
    cleaned_resumes = [clean_text(r["raw_text"]) for r in resume_data]
    
    sbert_model, sbert_ready = load_sbert_model() if use_sbert else (None, False)
    
    if use_sbert and sbert_ready:
        semantic_scores = compute_sbert_similarity(sbert_model, job_description, [r["raw_text"] for r in resume_data])
        engine_label = "🧠 Sentence-BERT (all-MiniLM-L6-v2)"
        weight_semantic = 0.50
        weight_skill = 0.40
        weight_keyword = 0.10
    else:
        # High-performance N-Gram TF-IDF
        corpus = [cleaned_jd] + cleaned_resumes
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            max_df=0.95,
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
        semantic_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        engine_label = "⚡ Fast Hybrid TF-IDF"
        weight_semantic = 0.55
        weight_skill = 0.35
        weight_keyword = 0.10
        
    results = []
    for idx, r in enumerate(resume_data):
        raw_text = r["raw_text"]
        resume_skills = extract_skills(raw_text)
        
        # Skill Match Metrics
        matched_skills = sorted(list(jd_skills.intersection(resume_skills)))
        missing_skills = sorted(list(jd_skills.difference(resume_skills)))
        
        if len(jd_skills) > 0:
            skill_score = len(matched_skills) / len(jd_skills)
        else:
            skill_score = float(semantic_scores[idx])
            
        jd_words = set(cleaned_jd.split())
        resume_words = set(cleaned_resumes[idx].split())
        keyword_overlap = len(jd_words.intersection(resume_words)) / max(len(jd_words), 1)
        
        sim_val = max(0.0, float(semantic_scores[idx]))
        composite_score = (sim_val * weight_semantic) + (skill_score * weight_skill) + (keyword_overlap * weight_keyword)
        final_percentage = round(min(composite_score * 100.0, 100.0), 2)
        
        if final_percentage >= 75:
            match_category = "🟢 Strong Match"
        elif final_percentage >= 50:
            match_category = "🟡 Moderate Match"
        else:
            match_category = "🔴 Low Match"
            
        results.append({
            "name": r["name"],
            "score": final_percentage,
            "semantic_score": round(sim_val * 100, 2),
            "skill_score": round(skill_score * 100, 2),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "total_skills_found": len(resume_skills),
            "word_count": len(raw_text.split()),
            "category": match_category,
            "engine_used": engine_label,
            "raw_text": raw_text
        })
        
    results.sort(key=lambda x: x["score"], reverse=True)
    return results, jd_skills, engine_label


# ---------------------------------------------------------
# UI Assets & Styling
# ---------------------------------------------------------
def get_base64_of_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return ""

bg_image_base64 = get_base64_of_image("background.png")
logo_image_base64 = get_base64_of_image("image.png")

bg_css = f'background: url("data:image/png;base64,{bg_image_base64}") no-repeat center center fixed; background-size: cover;' if bg_image_base64 else 'background-color: #0b0f19;'

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    
    .main {{
        {bg_css}
    }}
    
    .glass-card {{
        background: rgba(15, 23, 42, 0.80);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 22px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }}
    
    .glass-card:hover {{
        border-color: rgba(56, 189, 248, 0.55);
        transform: translateY(-2px);
    }}
    
    .metric-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }}
    
    .badge-green {{
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }}
    
    .badge-red {{
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }}
    
    .badge-blue {{
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }}
    
    .stButton>button {{
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        border: 1px solid #38bdf8;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.6);
        transform: scale(1.02);
    }}
    
    .hero-title {{
        text-align: center;
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }}
    
    .hero-subtitle {{
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Header & Hero Section
# ---------------------------------------------------------
col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
with col_logo2:
    if logo_image_base64:
        st.markdown(
            f'<div style="text-align:center; margin-bottom: 10px;"><img src="data:image/png;base64,{logo_image_base64}" width="110"></div>',
            unsafe_allow_html=True
        )
    st.markdown('<div class="hero-title">AI-Powered Resume Screener</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Intelligent candidate screening using Hybrid SBERT & Skill Extraction NLP</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Engine Selection, Presets & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.image("logo.png" if os.path.exists("logo.png") else "image.png" if os.path.exists("image.png") else None, width=120)
    
    st.markdown("### 🧠 AI Engine Selection")
    model_choice = st.radio(
        "Choose Screening AI Model:",
        options=["Sentence-BERT (Recommended)", "Fast Hybrid TF-IDF"],
        index=0,
        help="Sentence-BERT provides deep contextual semantics. Fast Hybrid TF-IDF is lightweight and instantaneous."
    )
    use_sbert = (model_choice == "Sentence-BERT (Recommended)")
    
    st.markdown("---")
    st.markdown("### ⚙️ Quick Presets")
    
    SAMPLE_JDS = {
        "Custom (Enter Your Own)": "",
        "Python / ML Engineer": (
            "We are seeking a Machine Learning Engineer proficient in Python, Scikit-Learn, PyTorch, or TensorFlow. "
            "Experience with NLP, Transformers, Large Language Models (LLMs), Pandas, NumPy, and Data Science workflows. "
            "Experience in Docker, Git, REST APIs, FastAPI, and Cloud Platforms (AWS/GCP/Azure) is highly desirable."
        ),
        "Full Stack Developer": (
            "Looking for a Full Stack Developer experienced with React, Next.js, TypeScript, Node.js, Express, and HTML5/CSS3. "
            "Must be skilled with SQL databases like PostgreSQL or MySQL, NoSQL with MongoDB, Redis caching, Git, and Docker. "
            "Familiarity with CI/CD pipelines, REST APIs, GraphQL, and Agile methodologies is required."
        ),
        "DevOps & Cloud Specialist": (
            "Hiring a DevOps Engineer with expertise in AWS, Kubernetes, Docker, Terraform, CI/CD with Jenkins or GitHub Actions. "
            "Strong background in Linux administration, Shell scripting, Python, Prometheus, Grafana, and microservices architecture."
        )
    }
    
    selected_preset = st.selectbox("Choose a Job Description Preset:", list(SAMPLE_JDS.keys()))
    
    st.markdown("---")
    st.markdown("### 📊 Model Scoring Architecture")
    if use_sbert:
        st.info(
            "**🧠 SBERT Hybrid Architecture:**\n\n"
            "• **Sentence-BERT (50%)**: Deep semantic context & synonym understanding.\n"
            "• **Skill Coverage (40%)**: Hard requirement verification.\n"
            "• **Keyword Overlap (10%)**: Domain requirement density."
        )
    else:
        st.info(
            "**⚡ Fast TF-IDF Architecture:**\n\n"
            "• **TF-IDF N-Grams (55%)**: Vocabulary & phrase matching.\n"
            "• **Skill Coverage (35%)**: Hard requirement verification.\n"
            "• **Keyword Overlap (10%)**: Broad terminology density."
        )
    
    st.markdown("---")
    st.markdown("🌐 **[Live Streamlit App](https://ai-resume-screening-sfdhusg7bdptmf7mwazwgz18.streamlit.app/)**")

# ---------------------------------------------------------
# Main Input Section
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown("### 📋 1. Job Description")
    initial_jd = SAMPLE_JDS[selected_preset] if selected_preset != "Custom (Enter Your Own)" else ""
    job_description = st.text_area(
        "Paste the Job Description or Requirements here:",
        value=initial_jd,
        height=220,
        placeholder="e.g. Seeking a Senior Data Scientist skilled in Python, NLP, Machine Learning, Docker, and SQL..."
    )
    
    extracted_jd_skills = extract_skills(job_description) if job_description.strip() else set()
    if extracted_jd_skills:
        st.markdown("**Detected Key Skills in JD:**")
        badges = " ".join([f'<span class="metric-badge badge-blue">{skill}</span>' for skill in sorted(extracted_jd_skills)])
        st.markdown(badges, unsafe_allow_html=True)

with col_right:
    st.markdown("### 📂 2. Upload Candidate Resumes")
    uploaded_files = st.file_uploader(
        "Upload Resumes (Supports PDF, DOCX, TXT):",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload multiple candidate resumes for automated batch screening and ranking."
    )
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} resumes ready for processing.")

# ---------------------------------------------------------
# File Previews (Collapsible Accordion)
# ---------------------------------------------------------
if uploaded_files:
    with st.expander("🔍 Preview Uploaded Resumes Content", expanded=False):
        preview_tabs = st.tabs([f"📄 {f.name[:20]}" for f in uploaded_files[:6]])
        for idx, tab in enumerate(preview_tabs):
            with tab:
                file = uploaded_files[idx]
                text = extract_resume_text(file)
                st.caption(f"**Filename:** {file.name} | **Words:** {len(text.split())} | **Characters:** {len(text)}")
                st.text_area(f"Preview: {file.name}", value=text[:1200] + ("..." if len(text) > 1200 else ""), height=150, key=f"prev_{idx}")

# ---------------------------------------------------------
# Processing & Ranking Section
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
process_button = st.button("🚀 Process & Rank Resumes", use_container_width=True)

if process_button:
    if not job_description.strip():
        st.error("⚠️ Please provide a Job Description to evaluate resumes against!")
    elif not uploaded_files:
        st.error("⚠️ Please upload at least one resume (PDF, DOCX, or TXT)!")
    else:
        engine_name = "Sentence-BERT" if use_sbert else "Fast TF-IDF"
        with st.spinner(f"🧠 Screening resumes using {engine_name} & Skill Extraction..."):
            resume_data = []
            for file in uploaded_files:
                text = extract_resume_text(file)
                resume_data.append({
                    "name": file.name,
                    "raw_text": text
                })
            
            results, jd_skills, active_engine = calculate_scores(job_description, resume_data, use_sbert=use_sbert)
            
            st.session_state["results"] = results
            st.session_state["jd_skills"] = list(jd_skills)
            st.session_state["active_engine"] = active_engine

# ---------------------------------------------------------
# Display Results & Visualizations
# ---------------------------------------------------------
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    jd_skills = set(st.session_state.get("jd_skills", []))
    active_engine = st.session_state.get("active_engine", "AI Model")
    
    st.markdown("---")
    st.markdown(f"## 📊 Screening Results & Candidate Rankings <span style='font-size:1rem; color:#38bdf8; font-weight:normal;'>({active_engine})</span>", unsafe_allow_html=True)
    
    # Overview Top Metrics
    top_candidate = results[0]
    avg_score = round(np.mean([r["score"] for r in results]), 1)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Resumes", len(results))
    with m_col2:
        st.metric("Top Candidate", top_candidate["name"][:18])
    with m_col3:
        st.metric("Highest Match Score", f"{top_candidate['score']}%")
    with m_col4:
        st.metric("Average Score", f"{avg_score}%")
    
    # -----------------------------------------------------
    # Top 3 Highlighting Cards
    # -----------------------------------------------------
    st.markdown("### 🏆 Top Matched Candidates")
    top_cols = st.columns(min(3, len(results)))
    
    for idx, col in enumerate(top_cols):
        cand = results[idx]
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <h3 style="margin-top:0; color:#38bdf8;">#{idx+1} {cand['name'][:22]}</h3>
                    <h1 style="margin: 5px 0; color:#ffffff; font-size:2.2rem;">{cand['score']}%</h1>
                    <p style="margin:0 0 10px 0;">{cand['category']}</p>
                    <p style="color:#94a3b8; font-size:0.9rem; margin-bottom:5px;">
                        <b>Semantic Match:</b> {cand['semantic_score']}% | <b>Skill Match:</b> {cand['skill_score']}%
                    </p>
                    <p style="color:#cbd5e1; font-size:0.85rem;">
                        <b>Matched Skills ({len(cand['matched_skills'])}):</b><br>
                        {', '.join(cand['matched_skills'][:5]) if cand['matched_skills'] else 'None detected'}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    # -----------------------------------------------------
    # Visual Analytics Chart
    # -----------------------------------------------------
    st.markdown("### 📈 Match Score Comparison")
    
    plot_df = pd.DataFrame(results)
    fig = px.bar(
        plot_df,
        x="score",
        y="name",
        orientation="h",
        color="score",
        color_continuous_scale=["#ef4444", "#eab308", "#22c55e", "#0284c7"],
        text="score",
        labels={"score": "Match Score (%)", "name": "Resume File"},
        title="Candidate Match Score Distribution"
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 100]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=max(350, len(results) * 45)
    )
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # -----------------------------------------------------
    # Detailed Candidate Breakdown & Skill Gap Table
    # -----------------------------------------------------
    st.markdown("### 🔬 Candidate Skill Gap & Match Details")
    
    for r in results:
        with st.expander(f"📌 {r['name']} — {r['score']}% ({r['category']})", expanded=(r == top_candidate)):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**Overall Score:** `{r['score']}%`")
                st.markdown(f"**Semantic Similarity:** `{r['semantic_score']}%`")
                st.markdown(f"**Skill Overlap:** `{r['skill_score']}%`")
                st.markdown(f"**Word Count:** `{r['word_count']}` words")
                st.markdown(f"**Model Engine:** `{r['engine_used']}`")
            with c2:
                st.markdown("**✅ Matched Skills:**")
                if r["matched_skills"]:
                    badges = " ".join([f'<span class="metric-badge badge-green">✓ {s}</span>' for s in r["matched_skills"]])
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.write("_No direct skill matches detected_")
                    
                st.markdown("<br>**❌ Missing Skills from JD:**", unsafe_allow_html=True)
                if r["missing_skills"]:
                    missing_badges = " ".join([f'<span class="metric-badge badge-red">✗ {s}</span>' for s in r["missing_skills"]])
                    st.markdown(missing_badges, unsafe_allow_html=True)
                else:
                    st.markdown('<span class="metric-badge badge-green">All JD skills present!</span>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # Export Report as CSV
    # -----------------------------------------------------
    export_df = pd.DataFrame([{
        "Rank": i + 1,
        "Resume Name": r["name"],
        "Final Match Score (%)": r["score"],
        "Semantic Score (%)": r["semantic_score"],
        "Skill Match Score (%)": r["skill_score"],
        "Category": r["category"],
        "Engine Used": r["engine_used"],
        "Matched Skills": ", ".join(r["matched_skills"]),
        "Missing Skills": ", ".join(r["missing_skills"]),
        "Word Count": r["word_count"]
    } for i, r in enumerate(results)])
    
    csv_data = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Screening Report (CSV)",
        data=csv_data,
        file_name="AI_Resume_Screening_Report.csv",
        mime="text/csv",
        use_container_width=True
    )
