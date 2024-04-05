import streamlit as st
from streamlit_chat import message
import tempfile
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers

# Caminhos para o FAISS DB e o modelo LLM
DB_FAISS_PATH = 'vectorstore/db_faiss'
LLM_MODEL_PATH = "llama-2-7b-chat.ggmlv3.q8_0.bin"

llm = None

def carregar_llm():
    """Carrega o modelo LLM se ainda não estiver carregado."""
    global llm
    if llm is None:
        llm = CTransformers(
            model=LLM_MODEL_PATH,
            model_type="llama",
            max_new_tokens=512,
            temperature=0.5
        )
    return llm

# Interface do Streamlit
st.title("Chat com CSV usando Llama2 🦙🦜")
st.markdown("<h3 style='text-align: center; color: white;'>Construído por <a href='https://github.com/AIAnytime'>AI Anytime com ❤️ </a></h3>", unsafe_allow_html=True)

arquivo_enviado = st.sidebar.file_uploader("Envie seus Dados", type="csv")

if arquivo_enviado:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(arquivo_enviado.getvalue())
        tmp_file_path = tmp_file.name

    carregador = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    dados = carregador.load()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(dados, embeddings)
    db.save_local(DB_FAISS_PATH)
    chain = ConversationalRetrievalChain.from_llm(llm=carregar_llm(), retriever=db.as_retriever())

    # Inicializa o estado da sessão
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'cache' not in st.session_state:
        st.session_state['cache'] = {}
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Olá! Pergunte-me qualquer coisa sobre " + arquivo_enviado.name + " 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Ei! 👋"]

    # Interface para o chat conversacional
    with st.container():
        with st.form(key='my_form'):
            entrada_usuario = st.text_input("Consulta:", placeholder="Fale com seus dados csv aqui (:", key='input')
            botao_enviar = st.form_submit_button(label='Enviar')

        if botao_enviar and entrada_usuario:
            saida = chat_conversacional(entrada_usuario)
            st.session_state['past'].append(entrada_usuario)
            st.session_state['generated'].append(saida)

    # Exibe as mensagens do chat
    with st.container():
        for i, generated_message in enumerate(st.session_state['generated']):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(generated_message, key=str(i), avatar_style="thumbs")

def chat_conversacional(consulta):
    """Gerencia a lógica do chat conversacional, incluindo o cache de respostas."""
    if consulta in st.session_state['cache']:
        return st.session_state['cache'][consulta]
    
    resultado = chain({"question": consulta, "chat_history": st.session_state['history']})
    st.session_state['history'].append((consulta, resultado["answer"]))
    st.session_state['cache'][consulta] = resultado["answer"]
    return resultado["answer"]
