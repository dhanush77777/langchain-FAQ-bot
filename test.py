import os
import openai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain import ElasticVectorSearch
from werkzeug.middleware.profiler import ProfilerMiddleware
from langchain.text_splitter import NLTKTextSplitter
import pandas as pd 
import pickle
import re
import redis
from fuzzywuzzy import fuzz        
from elasticsearch import Elasticsearch
from langchain.vectorstores.elastic_vector_search import ElasticKnnSearch
from langchain.embeddings import ElasticsearchEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
import sys

mappings = pickle.load(open('Answer_mappings.pkl','rb'))
df=pd.DataFrame(mappings)
with open('embedding_model.pkl', 'rb') as f:
    embedding = pickle.load(f)

opensearch_url = "https://localhost:9200",
http_auth = ("admin", "admin")

def similarity_search_batch(texts_batch, user_question):
    docsearch = OpenSearchVectorSearch.from_documents(
    texts_batch,
    embedding,
    opensearch_url=opensearch_url,
    http_auth=http_auth,
    use_ssl = False,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False,
    )
    return docsearch.similarity_search(user_question)
def get_content_after_string(string):
    search_string = "question_id:"
    index = string.find(search_string)
    
    if index == -1:
        return None
    
    content = string[index + len(search_string):].strip()
    return content
def f(t):
  results=similarity_search_batch(texts,t)           
  answer=get_content_after_string((results[0].page_content))
  return answer
loader = CSVLoader(file_path='Questions.csv',csv_args={'delimiter': ','})
data = loader.load()
text_splitter = NLTKTextSplitter(chunk_size=10000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
