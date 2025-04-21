import streamlit as st
import PyPDF2
from PIL import Image
import pytesseract
import requests
import logging

API_TOKEN = ""
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
            logging.FileHandler("app.log"),  # Log to a file
    ]
)
logger = logging.getLogger(__name__)

def pdf_to_text(uploaded_file):
    try:
        logger.info("Reading PDF file for text extraction.")
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        logger.info("PDF text extraction successful.")
        return text
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return f"Error reading PDF: {e}"


def image_to_text(uploaded_file):
    try:
        logger.info("Reading image file for OCR.")
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        logger.info("Image text extraction successful.")
        return text
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        return f"Error reading image: {e}"

def analyze_resume(job_description, resume_text):
    prompt = f"""
You are an expert recruiter.
Evaluate the following resume against the job description provided.

=== Job Description ===
{job_description}

=== Resume ===
{resume_text}

Please analyze and return the following:

1. Ovrall Match Score (0-100)
2. Category Ratings:
   - Skills Match (0-100)
   - Experience Relevance (0-100)
   - Project Alignment (0-100)
   - Education/Certifications (0-100)
3. Matched Skills, Tools, Technologies, Soft Skills
4. Missing or Weak Areas
5. Matching Projects
6. Qualifications & Certifications
7. Standout/Extraordinary Elements
8. Suggestions to Improve Resume

Return the response in readable Markdown format.
"""

    try:
        logger.info("Sending resume and job description to the model.")
        response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                full_text = result[0]["generated_text"]
                cleaned_output = full_text.replace(prompt.strip(), "").strip()
                logger.info("Resume analysis received successfully.")
                return cleaned_output
            else:
                logger.warning("Unexpected response format from model.")
                return "Unexpected response format."
        else:
            logger.error(f"API error {response.status_code}: {response.text}")
            return f"API error {response.status_code}: {response.text}"
    except Exception as e:
        logger.error(f"Failed to contact the model: {e}")
        return f"Failed to contact the model: {e}"


st.set_page_config(page_title="Resume Rating", layout="wide")
st.title("Resume Rating with Mistral LLM")

job_description = st.text_area("Job Description", height=300)
uploaded_file = st.file_uploader("Upload Resume (PDF or Image)", type=["pdf", "jpg", "jpeg"])

if uploaded_file and job_description:
    if st.button("Submit for Analysis"):
        file_name = uploaded_file.name.lower()
        logger.info(f"File uploaded: {file_name}")
        if file_name.endswith(".pdf"):
            resume_text = pdf_to_text(uploaded_file)
        else:
            resume_text = image_to_text(uploaded_file)

        if resume_text.startswith("❌"):
            st.error(resume_text)
        else:
            with st.spinner("Analyzing resume..."):
                analysis_result = analyze_resume(job_description, resume_text)

            st.markdown("---")
            st.subheader("Resume Analysis")
            st.markdown(analysis_result)
else:
    st.info("Write a job description and upload a resume to begin.")
