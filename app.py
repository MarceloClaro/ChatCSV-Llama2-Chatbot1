import streamlit as st
from htmlTemplates import css, bot_template, user_template
import pandas as pd
import os
import sys
import re
import tempfile
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def combine_csv_files_and_get_data(uploaded_files, output_path):
    combined_data = []
    data_list = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

            df = pd.read_csv(tmp_file.name)
            numeric_pattern = re.compile(r'\d+')
            year = numeric_pattern.search(tmp_file.name)

            # Add "year" column
            df["year"] = year

            # Append data to combined list
            combined_data.append(df)

    # Combine all dataframes into one
    combined_df = pd.concat(combined_data, ignore_index=True)

    # Save the combined data to a new CSV file
    combined_df.to_csv(output_path, index=False)

    print(f"Combined data saved to: {output_path}")

    # Loading a CSV file
    loader = CSVLoader(file_path = output_path, encoding = "utf-8", csv_args = {'delimiter': ','})
    data = loader.load()
    print("Data Loaded and returning the data for text chunking")
    return data

def get_text_chunks(final_data):
    # Splitting Data into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(final_data)
    print("Text Chunks Completed")
    return text_chunks

def get_embeddings(text_chunks, DB_FAISS_PATH):
    #Download Sentence Transformer embeddings from HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    docsearch = FAISS.from_documents(text_chunks, embeddings)
    docsearch.save_local(DB_FAISS_PATH)
    print(" --- Embeddings are stored into a vector database --- ")
    return docsearch

def load_llm(model_path):
    llm = CTransformers(
                    model= model_path,
                    model_type="llama",
                    config={
                        "max_new_tokens": 270,
                        "temperature": 0.1,
                    }
                )
    print("LLM loaded")
    return llm

def conversational_chat(chain,query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def main():
    DB_FAISS_PATH = "vectorstore/db_faiss/web_app2"

    # Choose the output destination
    output_path = "data/combined_data.csv"

    # Large Language Model Path
    llm_path = "C:/Projects/Project Langchain/models/llama-2-7b-chat.ggmlv3.q8_0.bin"

    st.set_page_config(page_title=" 📈 Chat with multiple CSVs 📈",
                       page_icon=":charts:")
    st.write(css, unsafe_allow_html=True)

    st.title("Chat with CSV using Llama2 🦙🦜")
    uploaded_files = st.sidebar.file_uploader("Upload your Data", accept_multiple_files=True, type="csv")

    if uploaded_files:
        data = combine_csv_files_and_get_data(uploaded_files, output_path)

        # get the text chunks
        text_chunks = get_text_chunks(data)

        # get embeddings
        docsearch = get_embeddings(text_chunks,DB_FAISS_PATH)

        # Import my language model
        llm = load_llm(llm_path)

        conversational_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_turns=5)

        chain = ConversationalRetrievalChain.from_llm(
                                                    llm,
                                                    retriever=docsearch.as_retriever(),
                                                    memory = conversational_memory
                                                )
        print("Chain and Memory activated")

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me anything about all data🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]
            
        #container for the chat history
        response_container = st.container()
        #container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
            
                user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
            
            if submit_button and user_input:
                output = conversational_chat(chain,user_input)
            
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")


if __name__ == '__main__':
        main()
