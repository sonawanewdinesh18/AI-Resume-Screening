# AI-Powered Resume Screening and Ranking System

# Overview
The AI-Powered Resume Screening and Ranking System is a web-based application that automatically
screens and ranks resumes based on a given job description using **Natural Language Processing (NLP) techniques.
The system utilizes TF-IDF (Term Frequency-Inverse Document Frequency) and Cosine Similarity to evaluate how well a resume matches a job description.

# Features
- Upload multiple resumes (PDF, DOCX, TXT formats)
- Extract text from resumes using *PyPDF2* and *python-docx*
- Enter a *job description*for comparison
- Rank resumes based on *TF-IDF and Cosine Similarity*
- Display results in a *sorted table*
- Highlight *Top 3 Best-Matching Resumes*
- Generate *bar chart visualization* of similarity scores
- Modern and *interactive UI using Streamlit*

# Technologies Used
- *Python* (Backend logic)
- *Streamlit* (Frontend UI)
- *Natural Language Processing (NLP)*
  - `sklearn.feature_extraction.text.TfidfVectorizer`
  - `sklearn.metrics.pairwise.cosine_similarity`
- *PyPDF2* & *python-docx* (Extract text from resumes)
- *Pandas* (Data manipulation)
- *Matplotlib & Seaborn* (Data visualization)

# Folder Structure
```
├── main.py                 # Main Streamlit app
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── sample_resumes/         # Folder to store sample resumes
├── assets/                 # Background and logo images
```

# Installation & Setup
#1️⃣ Clone the Repository
```
https://github.com/sonawanewdinesh18/AI-Resume-Screening
```

#2️⃣ Create a Virtual Environment (Optional but Recommended)
```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

#3️⃣ Install Required Packages
```
pip install -r requirements.txt
```

#4️⃣ Run the Streamlit App
```
streamlit run main.py
```

## How It Works
1. Upload Resumes (PDF, DOCX, or TXT format)
2. Enter Job Description in the text box
3. Click "Process Resumes" to analyze and rank them
4. View ranked resumes in a table and a bar chart visualization
5. The top 3 matching resumes are highlighted

# Deployment
You can deploy the project on *Streamlit Cloud*:
```
https://ai-resume-screening-sfdhusg7bdptmf7mwazwgz18.streamlit.app/
```
Then follow Streamlit Cloud's deployment guide.
Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Added new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a Pull Request

# License
This project is **open-source** and available under the *MIT License*.



