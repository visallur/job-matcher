import streamlit as st
import requests

st.title("Job–Resume Matcher")

job = st.text_area("Enter Job Description")

if st.button("Find Matches"):
    res = requests.post("http://127.0.0.1:8000/match", params={"job_description": job})
    st.write(res.json())