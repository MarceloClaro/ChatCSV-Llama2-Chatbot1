import streamlit as st 
from streamlit_chat import message
import tempfile

from langchain_community.document_loaders.csv_loader import CSVLoader


from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers


DB_FAISS_PATH = 'vectorstore/db_faiss'

# Carregando o modelo
def carregar_llm():
    # Carregar o modelo baixado localmente aqui
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

st.title("Chat com CSV usando Llama2 🦙🦜")
st.markdown("<h3 style='text-align: center; color: white;'>Construído por <a href='https://github.com/AIAnytime'>AI Anytime com ❤️ </a></h3>", unsafe_allow_html=True)

arquivo_enviado = st.sidebar.file_uploader("Envie seus Dados", type="csv")

if arquivo_enviado:
    # usar tempfile porque CSVLoader aceita apenas um file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(arquivo_enviado.getvalue())
        tmp_file_path = tmp_file.name

    carregador = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','})
    dados = carregador.load()
    # st.json(dados)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(dados, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm = carregar_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def chat_conversacional(consulta):
        resultado = chain({"question": consulta, "chat_history": st.session_state['history']})
        st.session_state['history'].append((consulta, resultado["answer"]))
        return resultado["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Olá! Pergunte-me qualquer coisa sobre " + arquivo_enviado.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Ei! 👋"]
        
    # container para o histórico do chat
    response_container = st.container()
    # container para a entrada de texto do usuário
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            entrada_usuario = st.text_input("Consulta:", placeholder="Fale com seus dados csv aqui (:", key='input')
            botao_enviar = st.form_submit_button(label='Enviar')
            
        if botao_enviar and entrada_usuario:
            saida = chat_conversacional(entrada_usuario)
            
            st.session_state['past'].append(entrada_usuario)
            st.session_state['generated'].append(saida)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
