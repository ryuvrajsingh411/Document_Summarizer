# Document Summarizer and Q&A Application

## Overview

This Streamlit application performs *Document Summarization* and *Question Answering from PDFs* using Hugging Face models. Users can upload PDF files to extract and process text, either summarizing the document or asking specific questions about its content.
You can Try it on [DocumentSummarizer](https://document-summarizer-keiqzmkogqq7czm4coyj95.streamlit.app/).

## Features
1. *Question Answering PDFs*:
   - Upload a PDF document.
   - Enter a question about the document to receive a concise answer.
   - View the history of questions and answers.

2. *Document Summarization*:
   - Upload a PDF document.
   - Enter a custom prompt to summarize the document.
   - Optionally specify a max_length in the prompt for summary length control.

## Dependencies

Ensure you have the following Python libraries installed:
- streamlit
- transformers
- PyPDF2
- langchain
- sentence-transformers
- faiss-cpu

To install dependencies, run:
terminal
pip install -r requirements.txt


## File Structure
- *main.py*: The primary script containing the app's code.
- *doc_file.pdf*: Temporary storage for the uploaded PDF file during processing.

## Setup

1. *Install Python*: Use Python 3.8 or later.

2. *Install Dependencies*:
   Install the required Python libraries as shown above.

3. *Hugging Face API Key*:
   - Obtain your API key from [Hugging Face](https://huggingface.co/settings/tokens).
   - Save the key in your secrets.toml file in the .streamlit directory:
     
     [secrets]
     HUGGINGFACEHUB_API_KEY = "your_api_key_here"
     

4. *Run the App*:
   Start the Streamlit app:
   bash
   streamlit run main.py
   

## How to Use

### 1. *Question Answering PDFs*
   - Select the "Question Answering PDFs" option from the sidebar.
   - Upload your PDF document.
   - Enter your question in the text box.
   - Click *Answer* to generate a response.
   - Review previous Q&A interactions under the "Question & Answer History" section.

### 2. *Summarize Document*
   - Select the "Summarize Document" option from the sidebar.
   - Upload your PDF document.
   - Enter a prompt to guide the summarization, e.g., "Summarize in 200 words".
   - Click *Summarize Document* to generate a concise summary.

## Important Notes
1. *Caching*: 
   - The app caches text extraction and embeddings to speed up repeated queries during a session.
2. *Max Token Length*:
   - The document is split into chunks of 1024 tokens with an overlap of 50 tokens for better processing.
3. *Embeddings*:
   - Uses sentence-transformers/all-MiniLM-L6-v2 for efficient vector representation.
4. *Language Model*:
   - Summarization uses MBZUAI/LaMini-Flan-T5-248M.
   - Question answering uses Mistral-7B-Instruct-v0.3.

## Customization
- Modify *appearance* by tweaking the CSS under st.markdown for a personalized look.
- Replace Hugging Face models with other compatible models in the AutoTokenizer, pipeline, or HuggingFaceEndpoint functions.

## Example Prompts
- *Question Answering*: 
   - "What is the total sales amount mentioned in the report?"
- *Summarization*: 
   - "Summarize the document focusing on financial data in 100 words."

## Troubleshooting
1. *File Processing Errors*:
   - Ensure the uploaded file is a valid PDF.
   - Check file permissions if errors persist.
2. *API Key Issues*:
   - Ensure your Hugging Face API key is correctly set in the secrets.toml file.
3. *Dependencies Missing*:
   - Reinstall missing dependencies using pip.
