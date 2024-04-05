import streamlit as st
from streamlit_chat import message
import tempfile
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_community.retrievers import VectorStoreRetriever

# Caminhos para o banco de dados FAISS e o modelo LLM
DB_FAISS_PATH = 'vectorstore/db_faiss'
LLM_MODEL_PATH = "llama-2-7b-chat.ggmlv3.q8_0.bin"

llm = None
chain = None

def load_llm():
    """Carrega o modelo LLM, se ainda não estiver carregado."""
    global llm
    if llm is None:
        llm = CTransformers(
            model=LLM_MODEL_PATH,
            model_type="llama",
            max_new_tokens=512,
            temperature=0.5
        )
    return llm

def setup_streamlit_interface():
    """Configura a interface do Streamlit."""
    st.title("🦙 Chat Inteligente com Dados CSV")
    st.markdown("Converse de forma interativa com seus dados CSV.")
    uploaded_file = st.sidebar.file_uploader("Envie seus Dados", type="csv")
    return uploaded_file

def process_uploaded_file(uploaded_file):
    """Processa o arquivo enviado."""
    global chain
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    retriever = VectorStoreRetriever(db)
    chain = ConversationalRetrievalChain.from_llm(llm=load_llm(), retriever=retriever)
    return chain

def conversational_chat(query):
    """Gerencia a lógica do chat conversacional, incluindo o cache de respostas."""
    global chain
    if query in st.session_state['cache']:
        return st.session_state['cache'][query]
    
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    st.session_state['cache'][query] = result["answer"]
    return result["answer"]

def main():
    uploaded_file = setup_streamlit_interface()
    if uploaded_file:
        chain = process_uploaded_file(uploaded_file)
        st.session_state.setdefault('history', [])
        st.session_state.setdefault('cache', {})
        st.session_state.setdefault('generated', ["Olá! Pergunte-me qualquer coisa sobre " + uploaded_file.name + " 🤗"])
        st.session_state.setdefault('past', ["Oi! 👋"])

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

if __name__ == "__main__":
    main()
