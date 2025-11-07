# medical_chatbot.py

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()

PDF_PATH = "data/medical_docs"

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

# === OpenAI LLM ===
llm = ChatOpenAI(
    model="gpt-4o-mini",   # or gpt-4-turbo, gpt-3.5-turbo
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# === Prompt Template ===
template = """
<|context|>
You are a helpful medical assistant that follows the instructions and generates accurate, concise responses based on the query and context provided.
Please be truthful and direct.
</s>
<|user|>
{query}
</s>
<|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)

# === RAG Chain ===
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def get_response(query: str):
    """Helper function for Streamlit."""
    return rag_chain.invoke(query)


