import os

import openai
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
)
from dotenv import load_dotenv

load_dotenv()

#accessing openapi key
#openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings() # type: ignore


def create_db_from_youtube_video(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)  #to load data from the video url.
    transcript = loader.load()
    #print(transcript)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  #this loaded data would be huge, so splitting the same in many diff docs.
    document = text_splitter.split_documents(transcript)
    #print(document)

    db = FAISS.from_documents(document, embeddings)
    return db

video_url = "https://www.youtube.com/watch?v=MiVp4wFlpjQ"
data_base = create_db_from_youtube_video(video_url)