import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import docx
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    """Extracts text from an uploaded DOCX file."""
    text = ""
    try:
        doc = docx.Document(docx_file)
        text = "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
    return text

# Function to encode an image to base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load offline images
bg_image_path = "background.png"
logo_image_path = "image.png"

bg_image_base64 = get_base64_of_image(bg_image_path)
logo_image_base64 = get_base64_of_image(logo_image_path)

# Streamlit UI Styling with Hover Effects
st.markdown(
    f"""
    <style>
    .main {{
        background: url("data:image/png;base64,{bg_image_base64}") no-repeat center center fixed;
        background-size: cover;
    }}
    .centered-image {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }}

    /* Heading Hover Effects */
    h1, h2, h3 {{
        font-family: 'Arial', sans-serif;
        color: #0077b6;
        text-align: center;
        transition: 0.3s ease-in-out;
        padding-bottom: 5px;
        position: relative;
    }}
    h1::after, h2::after, h3::after {{
        content: "";
        display: block;
        width: 0;
        height: 3px;
        background: #00a8e8;
        transition: width 0.3s ease-in-out;
        margin: auto;
    }}
    h1:hover::after, h2:hover::after, h3:hover::after {{
        width: 50%;
    }}
    h1:hover, h2:hover, h3:hover {{
        color: #00a8e8;
        text-shadow: 0px 4px 10px rgba(0, 168, 232, 0.5);
    }}

    /* Button Effects */
    .stButton>button {{
        background-color: #0077b6;
        color: white;
        border-radius: 20px;
        padding: 12px 25px;
        font-weight: bold;
        transition: 0.3s;
        border: 2px solid #00a8e8;
        box-shadow: 0px 4px 10px rgba(0, 168, 232, 0.5);
    }}
    .stButton>button:hover {{
        background-color: #00a8e8;
        border-color: #0077b6;
        transform: scale(1.05);
    }}

    /* Resume Hover Effect */
    .result-card {{
        background: rgba(255, 255, 255, 0.85);
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0px 5px 15px rgba(0, 168, 232, 0.5);
        margin-bottom: 10px;
        backdrop-filter: blur(8px);
        transition: transform 0.3s, box-shadow 0.3s;
    }}
    .result-card:hover {{
        transform: scale(1.03);
        box-shadow: 0px 5px 20px rgba(0, 168, 232, 0.7);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("🧠 AI-Powered Resume Screener")
st.markdown(f'<div class="centered-image"><img src="data:image/png;base64,{logo_image_base64}" width="150"></div>', unsafe_allow_html=True)

st.write("## Upload Resumes")
uploaded_files = st.file_uploader("Upload Resumes (PDF, DOCX, TXT)", accept_multiple_files=True, type=["pdf", "docx", "txt"])

st.write("## Job Description")
job_description = st.text_area("Enter Job Description", height=150)

if uploaded_files:
    st.write("## 🔍 File Previews")
    for file in uploaded_files:
        file_type = file.type
        if file_type == "application/pdf":
            st.write(f"#### 📄 Preview: {file.name}")
            pdf_text = extract_text_from_pdf(file)
            st.text_area("Content", pdf_text[:1000], height=200)

        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            st.write(f"#### 📝 Preview: {file.name}")
            doc_text = extract_text_from_docx(file)
            st.text_area("Content", doc_text[:1000], height=200)

        elif file_type == "text/plain":
            st.write(f"#### 📜 Preview: {file.name}")
            txt_text = str(file.read(), "utf-8")
            st.text_area("Content", txt_text[:1000], height=200)

    st.info(f"📄 {len(uploaded_files)} resumes uploaded")

# Process Resumes and Rank Them
if st.button("Process Resumes"):
    if uploaded_files and job_description:
        resume_texts = [extract_text_from_pdf(file) if file.type == "application/pdf"
                         else extract_text_from_docx(file) if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                         else str(file.read(), "utf-8")
                         for file in uploaded_files]

        texts = [job_description] + resume_texts
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        results = sorted(zip([file.name for file in uploaded_files], scores), key=lambda x: x[1], reverse=True)
        
        st.write("## Matching Results")
        df = pd.DataFrame(results, columns=["Resume Name", "Score"])
        st.dataframe(df)

        st.write("### 🌟 Top 3 Best Matching Resumes")
        for idx, (name, score) in enumerate(results[:3]):
            st.markdown(
                f"""
                <div class="result-card">
                    <b>{idx+1}. {name}</b><br>
                    Match Score: <b>{score:.2f}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plot Matching Scores
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(y=[r[0] for r in results[:10]], x=[r[1] for r in results[:10]], palette="Blues", ax=ax)
        ax.set_xlabel("Score")
        ax.set_ylabel("Resume Name")
        ax.set_title("Top Matching Resumes")
        st.pyplot(fig)

    else:
        st.error("⚠️ Please upload resumes and provide a job description!")
