import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

def load_llm():
    """
    Carrega o modelo de linguagem localmente.
    """
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def process_uploaded_file(uploaded_file):
    """
    Processa o arquivo CSV enviado pelo usuário.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    return data

def setup_chat():
    """
    Configura o ambiente de chat.
    """
    st.title("🦙Chat com CSV usando Llama2🦜")
    st.markdown("<h3 style='text-align: center; color: white;'></a></h3>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Carregar seus Dados", type="csv")

    if uploaded_file:
        data = process_uploaded_file(uploaded_file)
        return data
    else:
        return None

def main():
    """
    Função principal para executar o aplicativo Streamlit.
    """
    data = setup_chat()

    if data:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

        db = FAISS.from_documents(data, embeddings)
        db.save_local(DB_FAISS_PATH)

        llm = load_llm()
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Olá! Pergunte-me qualquer coisa sobre " + uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Ei! 👋"]

        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Pergunta:", placeholder="Converse com seus dados CSV aqui (:", key='input')
                submit_button = st.form_submit_button(label='Enviar')

            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

if __name__ == "__main__":
    main()
