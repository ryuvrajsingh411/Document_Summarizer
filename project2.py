import streamlit as st
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import pipeline
from PyPDF2 import PdfReader
from langchain import LLMChain
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import GPT2Tokenizer
import re
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema.runnable import RunnablePassthrough


tokenizer2 = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
model = T5ForConditionalGeneration.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

api_key = st.secrets["HUGGINGFACEHUB_API_KEY"]


st.set_page_config(layout="wide", page_title="Document Summarizer", page_icon="📝")

st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(to right, #e0c3fc, #8ec5fc, #fbc2eb, #a6c1ee);
        color: #333;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #6a11cb, #2575fc);
        color: white;
    }
    .stButton button {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
        border-radius: 4px;
        border: none;
        padding: 10px 20px;
        cursor: pointer;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton button:hover {
        background: linear-gradient(to right, #2575fc, #6a11cb);
    }
    .stTextArea textarea {
        background-color: #ffffff;
        color: #333;
        border-radius: 4px;
        border: 1px solid #ced4da;
        padding: 10px;
        font-size: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stFileUploader label {
        color: #6a11cb;
        font-size: 16px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #343a40;
    }
    .stInfo {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 20px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .title h1 {
        font-size: 2.5em;
        background: -webkit-linear-gradient(#6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .title h2 {
        font-size: 2em;
        background: -webkit-linear-gradient(#6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def text_summary(text, temp, max_len=512):
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, max_length=max_len)
    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})
    final_prompt_template = f"""
                      {temp}.
                      ```{"{text}"}```
                   """

    template = """
        Write a concise and brief summary of the following text.

        ```{text}```

    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    prompt2 = PromptTemplate(template=final_prompt_template, input_variables=["text"])
    llm_chain2 = LLMChain(prompt=prompt2, llm=llm)

    result = ''
    for i in range(len(text)):
        result += llm_chain.run(text[i])

    result2 = llm_chain2.run(result)
    separated_lines = result2.split(' -')
    combined_string = '\n'.join(separated_lines)

    return combined_string


def get_conversation_chain(text_chunks, embeddings):

    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=api_key,
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        timeout=480
    )

    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use 3 sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain


def extract_text_from_pdf(file_path):
    # Open the PDF file using PyPDF2
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        page = reader.pages
        input_text = ''
        for i in range(len(page)):
            input_text += page[i].extract_text()

    return input_text

def spliter(input_text):
    tokens = tokenizer2.encode(input_text)
    max_length = 1024
    overlap = 50

    # Split tokens into chunks
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length-overlap)]

    # Decode chunks back to text if needed
    chunks_text = [tokenizer2.decode(chunk) for chunk in chunks]

    return chunks_text

def qapdf(text, query):
    chunk = spliter(text)
    chain = get_conversation_chain(chunk, embeddings)
    response = chain.invoke(query)
    return response

def extract_integer(text):
    # Find all integers in the text
    integers = re.findall(r'\d+', text)
    # Convert the first found integer to an integer type (if any found)
    return int(integers[0]) if integers else None

# Sidebar and main layout
st.sidebar.title("IITI SOC PROJECT")
choice = st.sidebar.selectbox("Select your choice", ["Question Answering PDFs", "Summarize Document"])

if choice == "Question Answering PDFs":
    st.markdown('<div class="title"><h1>Question Answering PDFs</h1></div>', unsafe_allow_html=True)
    input_file = st.file_uploader("Upload your document here", type=['pdf'])
    if input_file is not None:
        st.info("File uploaded successfully")

        if 'text' not in st.session_state:
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())
                st.session_state.text = extract_text_from_pdf("doc_file.pdf")
                st.session_state.history = []

        input_text = st.text_area("Enter your question here")
        if st.button("Answer"):
            result = qapdf(st.session_state.text, input_text)
            st.success(result)
            st.session_state.history.append({'question': input_text, 'answer': result})

        st.markdown("### Question & Answer History")
        for i, qa_pair in enumerate(st.session_state.history):
            st.markdown(f"**Question{i + 1}:** {qa_pair['question']}")
            st.markdown(f"**Answer{i + 1}:** {qa_pair['answer']}")

elif choice == "Summarize Document":
    st.markdown('<div class="title"><h1>Document Summarization</h1></div>', unsafe_allow_html=True)
    input_file = st.file_uploader("Upload your document here", type=['pdf'])
    if input_file is not None:
        st.info("File uploaded successfully")
        input_text = st.text_area("Enter your prompt here")
        max_len = extract_integer(input_text)
        if st.button("Summarize Document"):
            with open("doc_file.pdf", "wb") as f:
                f.write(input_file.getbuffer())

                st.markdown("*Summary Result*")
                text = extract_text_from_pdf("doc_file.pdf")
                input_l = spliter(text)
                if max_len is not None:
                    doc_summary = text_summary(input_l, input_text, max_len)
                else:
                    doc_summary = text_summary(input_l, input_text)
                st.success(doc_summary)
