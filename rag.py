import numpy as np
import pathlib
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

def get_vector_store():
  
  document = PyPDFLoader(r"C:\Users\USER\Downloads\Image_classification_on_Animals___Final_Report__Draft_1___2_.pdf").load()

  splits = RecursiveCharacterTextSplitter(
    chunk_size = 500, chunk_overlap = 200).split_documents(document)

  embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

  vectors_store = FAISS.from_documents(splits, embeddings)

  return vectors_store

vectors_store = get_vector_store()

 
def get_policy_response(query : str, vectorstore):

  llm = ChatGroq(
    model = 'qwen/qwen3.6-27b',
    temperature = 0
  )

  docs = vectorstore.similarity_search(query, k=3)

  context = "\n\n".join([doc.page_content for doc in docs])

  prompt = f"""
  You are a doc assistant and right now. You can ONLY use the context below to answer the question.

  Context:
  {context}

  Question:
  {query}

  """

  response = llm.invoke(prompt)

  return response

ask = input("Your queston: ")

response = get_policy_response(query = ask, vectorstore=vectors_store)

print(response.content)