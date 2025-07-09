import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import openai

# Initialize models
jd_parser = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Title
st.set_page_config("AI Recruiter Assistant", layout="wide")
st.title("🤖 AI Recruitment Assistant")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["JD Parser", "Resume Matcher", "Outreach", "Summary"])

# 1. JD Parser
with tab1:
    st.subheader("📄 Paste Job Description")
    jd_text = st.text_area("Job Description", height=200)
    if st.button("Extract Info from JD"):
        labels = ["Skills", "Years of Experience", "Education", "Job Title", "Location", "Industry"]
        result = jd_parser(jd_text, labels)
        st.json(dict(zip(result["labels"], result["scores"])))

# 2. Resume Matcher
with tab2:
    st.subheader("📌 Resume Matcher")
    jd_input = st.text_area("Paste JD here:", height=150)
    resume_input = st.text_area("Paste Resume here:", height=150)
    if st.button("Match Resume"):
        jd_emb = sbert_model.encode(jd_input, convert_to_tensor=True)
        resume_emb = sbert_model.encode(resume_input, convert_to_tensor=True)
        score = util.pytorch_cos_sim(jd_emb, resume_emb)
        st.metric("Match Score", round(float(score), 2))

# 3. Outreach Generator
with tab3:
    st.subheader("✉️ Generate Outreach Message")
    name = st.text_input("Candidate Name")
    role = st.text_input("Role Title")
    company = st.text_input("Company Name")
    recruiter = st.text_input("Your Name")
    if st.button("Generate Message"):
        prompt = f"Write a short LinkedIn outreach to {name} for the role of {role} at {company}, signed by {recruiter}."
        response = openai.ChatCompletion.create(model="gpt-4",
                        messages=[{"role": "user", "content": prompt}])
        st.write(response.choices[0].message['content'])

# 4. Candidate Summary
with tab4:
    st.subheader("🗂️ Summarize Resume")
    resume_text = st.text_area("Paste Resume here:", height=200)
    if st.button("Summarize"):
        prompt = f"Summarize this resume for a client:\n\n{resume_text}"
        response = openai.ChatCompletion.create(model="gpt-4",
                        messages=[{"role": "user", "content": prompt}])
        st.write(response.choices[0].message['content'])
