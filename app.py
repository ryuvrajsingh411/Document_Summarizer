import streamlit as st
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import pipeline
from PyPDF2 import PdfReader

tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
model = T5ForConditionalGeneration.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
pipe = pipeline("summarization", model=model, tokenizer=tokenizer)

st.set_page_config(layout="wide")


@st.cache_resource
def text_summary(text, maxlength=None):

    result = ''
    for i in range(len(text)):
        result += pipe(text[i])[0]['summary_text']
    return result


def extract_text_from_pdf(file_path):
    # Open the PDF file using PyPDF2
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages
        text = []
        for i in range(len(page)):
            text.append(page[i].extract_text())
    return text



choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])

if choice == "Summarize Text":
    st.subheader("Summarize Text")
    input_text = st.text_area("Enter your text here")
    if input_text is not None:
        if st.button("Summarize Text"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("*Your Input Text*")
                st.info(input_text)
            with col2:
                st.markdown("*Summary Result*")
                result = pipe(input_text)[0]['summary_text']
                st.success(result)

elif choice == "Summarize Document":
    st.subheader("Summarize Document")
    input_file = st.file_uploader("Upload your document here", type=['pdf'])
    if input_file is not None:
        if st.button("Summarize Document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
            col1, col2 = st.columns([1,1])
            with col1:
                st.info("File uploaded successfully")
                extracted_text = extract_text_from_pdf("doc_file.pdf")
                final_text = list_to_text(extracted_text)
                st.markdown("*Extracted Text is Below:*")
                st.info(final_text)
            with col2:
                st.markdown("*Summary Result*")
                text = extract_text_from_pdf("doc_file.pdf")
                doc_summary = text_summary(text)
                st.success(doc_summary)
