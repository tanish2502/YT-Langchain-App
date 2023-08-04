import os

import streamlit as st
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
import textwrap

load_dotenv()
embeddings = OpenAIEmbeddings() # type: ignore

def create_db_from_youtube_video(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)  #to load data from the video url.
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  #this loaded data would be huge, so splitting the same in many diff docs.
    document = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(document, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    query_relevant_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name ="gpt-3.5-turbo", temperature=0.9) # type: ignore

    system_template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(question=query, docs=query_relevant_content)
    response = response.replace("\n", "")
    return response


st.title('🦜️🔗 Youtube GPT Assistant')
video_url = st.text_input('Enter Youtube Video URL here: ')

#video_url = "https://www.youtube.com/watch?v=MiVp4wFlpjQ"
if video_url:
    data_base = create_db_from_youtube_video(video_url)
    st.write(data_base)

# query = "What are they saying about AGI and gpt4?"
# response = get_response_from_query(data_base, query)
# print(textwrap.fill(response, width=100))