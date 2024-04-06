import streamlit as st
from streamlit_chat import message
import pandas as pd
import requests
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_community.retrievers import VectorStoreRetriever

# Definições de caminho
DB_FAISS_PATH = 'vectorstore/db_faiss'
LLM_MODEL_PATH = "llama-2-7b-chat.ggmlv3.q8_0.bin"

llm = None
chain = None
prompt_initial = "Olá! Pergunte-me qualquer coisa sobre seus dados CSV 🤗"

def load_llm():
    global llm
    if llm is None:
        llm = CTransformers(
            model=LLM_MODEL_PATH,
            model_type="llama",
            max_new_tokens=512,
            temperature=0.5
        )
    return llm

def convert_csv_to_text(data):
    text_data = []
    for _, row in data.iterrows():
        text_row = ' '.join(str(val) for val in row)
        text_data.append(text_row)
    return text_data

def process_uploaded_file(uploaded_file):
    global chain
    csv_data = pd.read_csv(uploaded_file)
    text_data = convert_csv_to_text(csv_data)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(text_data, embeddings)
    db.save_local(DB_FAISS_PATH)
    retriever = VectorStoreRetriever(db)
    chain = ConversationalRetrievalChain.from_llm(llm=load_llm(), retriever=retriever)
    return chain

def setup_config_page():
    st.title("Configurações do Chatbot")
    st.subheader("Defina o prompt inicial:")
    global prompt_initial
    prompt_initial = st.text_area("Prompt Inicial", value=prompt_initial)
    st.session_state['prompt_initial'] = prompt_initial
    
    st.subheader("Baixar Dados CSV:")
    csv_url = st.text_input("URL do Arquivo CSV:")
    download_button = st.button("Baixar CSV")

    if download_button and csv_url:
        csv_file = download_csv_data(csv_url)
        if csv_file and validate_csv_file(csv_file):
            st.success(f"Arquivo {csv_file} baixado com sucesso!")
            st.session_state['chain'] = process_uploaded_file(csv_file)

def setup_chat_page():
    st.title("🦙 Chat Inteligente com Dados CSV")
    if 'chain' in st.session_state and st.session_state['chain']:
        with st.container():
            user_input = st.text_input("Pergunta:", placeholder="Converse aqui com os seus dados CSV:")
            send_button = st.button('Enviar')

            if send_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

            for i, generated_message in enumerate(st.session_state['generated']):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(generated_message, key=str(i))
    else:
        st.error("Nenhum modelo de chat carregado. Por favor, carregue seus dados CSV na página de configurações.")

def conversational_chat(query):
    global chain
    if query in st.session_state['cache']:
        return st.session_state['cache'][query]
    
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    st.session_state['cache'][query] = result["answer"]
    return result["answer"]

def download_csv_data(csv_url):
    try:
        r = requests.get(csv_url, stream=True)
        if r.status_code == 200:
            csv_filename = csv_url.split('/')[-1]
            with open(csv_filename, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                dl = 0
                for data in r.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    st.progress(dl / total_length)
            return csv_filename
        else:
            st.error("Erro ao baixar o arquivo. Verifique a URL.")
    except Exception as e:
        st.error(f"Erro ao baixar o arquivo: {e}")

def validate_csv_file(file_path):
    try:
        pd.read_csv(file_path)
        return True
    except Exception as e:
        st.error(f"Erro ao processar o arquivo CSV: {e}")
        return False

def main():
    st.sidebar.title("Navegação")
    app_mode = st.sidebar.radio("Escolha a página:", ["Configurações", "Chat"])
    
    if app_mode == "Configurações":
        setup_config_page()
        uploaded_file = st.file_uploader("Envie seus Dados CSV", type="csv")
        if uploaded_file is not None:
            st.session_state['chain'] = process_uploaded_file(uploaded_file)
            st.success("Dados CSV processados e chatbot carregado!")
    elif app_mode == "Chat":
        setup_chat_page()

if __name__ == "__main__":
    if 'generated' not in st.session_state:
        st.session_state['generated'] = [prompt_initial]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Usuário"]
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'cache' not in st.session_state:
        st.session_state['cache'] = {}
    if 'chain' not in st.session_state:
        st.session_state['chain'] = None
    main()
