# medical_chatbot.py

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCppcls
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import os

# === Paths ===
from dotenv import load_dotenv
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
PDF_PATH = os.getenv("PDF_PATH")


# === Load Documents ===
loader = PyPDFDirectoryLoader(PDF_PATH)
docs = loader.load()

# === Split Text ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# === Embeddings and Vector Store ===
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# === Load LLM ===
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.25,
    max_tokens=2048,
    top_p=1,
    verbose=False
)

# === Prompt Template ===
template = """
<|context|>
You are a helpful medical assistant that follows the instructions and generates accurate responses based on the query and context provided.
Please be truthful and give direct answers.
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)

# === Build RAG Chain ===
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def get_response(query: str):
    """Helper function for Streamlit."""
    return rag_chain.invoke(query)
