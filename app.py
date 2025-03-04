import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text.strip()

# Streamlit UI
st.title("AI-Powered Resume Screening & Ranking System")

# Upload Job Description
st.subheader("Upload Job Description (Text File)")
job_desc_file = st.file_uploader("Upload Job Description", type=["txt"])

# Upload Resumes
st.subheader("Upload Resumes (PDFs)")
resume_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

if job_desc_file and resume_files:
    # Read Job Description
    job_description = job_desc_file.read().decode("utf-8")

    # Extract text from resumes
    resume_data = []
    for resume in resume_files:
        resume_text = extract_text_from_pdf(resume)
        resume_data.append({"Name": resume.name, "Text": resume_text})

    # Convert to DataFrame
    df = pd.DataFrame(resume_data)

    # Combine Job Description with Resumes for Vectorization
    documents = [job_description] + df["Text"].tolist()

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute Cosine Similarity (Job Description vs. Resumes)
    job_vec = tfidf_matrix[0]  # Job Description vector
    resume_vecs = tfidf_matrix[1:]  # Resume vectors
    scores = cosine_similarity(job_vec, resume_vecs)[0]

    # Add similarity scores to DataFrame
    df["Match Score"] = scores * 100  # Convert to percentage
    df = df.sort_values(by="Match Score", ascending=False)

    # Display Results
    st.subheader("Ranked Resumes")
    st.dataframe(df[["Name", "Match Score"]])

    # Download Results as CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results as CSV", data=csv, file_name="ranked_resumes.csv", mime="text/csv")
    