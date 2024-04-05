from streamlit import title, markdown, sidebar, file_uploader, container, form_submit_button, text_input
from streamlit_chat import message
import tempfile
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers

DB_FAISS_PATH = 'vectorstore/db_faiss'
LLM_MODEL_PATH = "llama-2-7b-chat.ggmlv3.q8_0.bin"

# Carregando o modelo LLM
llm = None

def carregar_llm():
    global llm
    if llm is None:
        llm = CTransformers(
            model=LLM_MODEL_PATH,
            model_type="llama",
            max_new_tokens=512,
            temperature=0.5
        )
    return llm

title("Chat com CSV usando Llama2 🦙🦜")
markdown("<h3 style='text-align: center; color: white;'>Construído por <a href='https://github.com/AIAnytime'>AI Anytime com ❤️ </a></h3>", unsafe_allow_html=True)

arquivo_enviado = sidebar.file_uploader("Envie seus Dados", type="csv")

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

    def chat_conversacional(consulta):
        if consulta in st.session_state['cache']:
            return st.session_state['cache'][consulta]
        
        resultado = chain({"question": consulta, "chat_history": st.session_state['history']})
        st.session_state['history'].append((consulta, resultado["answer"]))
        st.session_state['cache'][consulta] = resultado["answer"]
        return resultado["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    if 'cache' not in st.session_state:
        st.session_state['cache'] = {}

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Olá! Pergunte-me qualquer coisa sobre " + arquivo_enviado.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Ei! 👋"]
        
    response_container = container()
    form_key = 'my_form'

    with container():
        with form_submit_button(key=form_key, label='Enviar') as botao_enviar:
            entrada_usuario = text_input("Consulta:", placeholder="Fale com seus dados csv aqui (:", key='input')
            
        if botao_enviar and entrada_usuario:
            saida = chat_conversacional(entrada_usuario)
            st.session_state['past'].append(entrada_usuario)
            st.session_state['generated'].append(saida)

    if st.session_state['generated']:
        with response_container:
            for i, generated_message in enumerate(st.session_state['generated']):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(generated_message, key=str(i), avatar_style="thumbs")
