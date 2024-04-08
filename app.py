import streamlit as st
import tempfile
import openai
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Set your OpenAI API key here
OPENAI_API_KEY = "your_openai_api_key"
openai.api_key = OPENAI_API_KEY

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    return tmp_file_path

def setup_chat():
    st.title("🦙 Chat com CSV usando Llama2 e OpenAI 🦜")
    st.markdown("<h3 style='text-align: center; color: white;'></a></h3>", unsafe_allow_html=True)
    
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.sidebar.write("Faça upload do arquivo CSV:")
    uploaded_file = st.sidebar.file_uploader("Carregar seus Dados", type="csv", accept_multiple_files=False, key='csv')

    if uploaded_file:
        data_path = process_uploaded_file(uploaded_file)
        return data_path
    else:
        return None

def chat_with_openai(user_input):
    response = openai.Completion.create(
      engine="davinci",
      prompt=user_input,
      temperature=0.7,
      max_tokens=150,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    return response.choices[0].text.strip()

def main():
    data_path = setup_chat()

    if data_path:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

        db = FAISS.from_documents(data_path, embeddings)
        db.save_local(DB_FAISS_PATH)

        llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", max_new_tokens=512, temperature=0.5)
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Olá! Pergunte-me qualquer coisa sobre o arquivo CSV 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Ei! 👋"]

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Pergunta:", placeholder="Converse com seus dados CSV aqui (:", key='input')
                submit_button = st.form_submit_button(label='Enviar')

            if submit_button and user_input:
                output = chat_with_openai(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    st.write(st.session_state["past"][i])
                    st.write(st.session_state["generated"][i])

if __name__ == "__main__":
    main()
