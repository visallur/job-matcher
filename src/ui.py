import streamlit as st
import os
from dotenv import load_dotenv


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from src.search import JobMatcher

load_dotenv()

st.set_page_config(page_title="AI Job Matcher", layout="wide")

@st.cache_resource
def load_matcher():
    return JobMatcher()

matcher = load_matcher()

def analyze_gaps(job_desc, resume_text):
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return "Error: OpenAI API Key not found. Please check your .env file."
    
    try:
        chat = ChatOpenAI( model_name="llama-3.3-70b-versatile", openai_api_key=api_key, openai_api_base="https://api.groq.com/openai/v1"  )
        
        prompt = f"""
        You are an expert Technical Recruiter. Compare the following Job Description and Candidate Resume.
        
        JOB DESCRIPTION:
        {job_desc}
        
        RESUME:
        {resume_text}
        
        Provide a concise analysis in this format:
        1. Match Score (0-100% based on skills):
        2. Key Strength (1 sentence):
        3. Critical Missing Skills (Bullet points):
        4. Final Recommendation (Interview or Reject):
        """
        
        messages = [
            SystemMessage(content="You are a helpful HR assistant."),
            HumanMessage(content=prompt)
        ]
        
        response = chat.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"

#UI Layout
st.title("Resume Matcher & Gap Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    job_input = st.text_area("Paste Job Description here:", height=300)
    
    if st.button("Find Matching Resumes"):
        if job_input:
            with st.spinner("Searching vector database..."):
                results = matcher.search_resumes_for_query(job_input, k=5)
            st.session_state['results'] = results
        else:
            st.warning("Please enter a job description.")

with col2:
    st.subheader("Top Candidates")
    
    if 'results' in st.session_state:
        for res in st.session_state['results']:
            # Safe access to category
            category = res.get('category', 'Candidate')
            
            with st.expander(f"Rank {res['rank']} - Score: {res['score']} - {category}"):
                st.write("**Resume Snippet:**")
                st.info(res['text'])
                btn_key = f"analyze_{res['rank']}"
                if st.button(f"Generate Gap Analysis (AI)", key=btn_key):
                    with st.spinner("Consulting Grok"):
                        analysis = analyze_gaps(job_input, res['text'])
                        st.markdown("### AI Assessment")
                        st.write(analysis)
st.sidebar.markdown("---")
st.sidebar.header("System Stats")
st.sidebar.text(f"Jobs Indexed: {len(matcher.jobs_df)}")
st.sidebar.text(f"Resumes Indexed: {len(matcher.resumes_df)}")