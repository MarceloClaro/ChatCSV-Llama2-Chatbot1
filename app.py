import streamlit as st
from streamlit_chat import message
import tempfile
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers

# Paths for the FAISS DB and the LLM model
DB_FAISS_PATH = 'vectorstore/db_faiss'
LLM_MODEL_PATH = "llama-2-7b-chat.ggmlv3.q8_0.bin"

llm = None

def load_llm():
    """Loads the LLM model if it's not already loaded."""
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
    """Sets up the Streamlit interface."""
    st.title("Chat with CSV using Llama2 🦙🦜")
    st.markdown("<h3 style='text-align: center; color: white;'>Built by <a href='https://github.com/AIAnytime'>AI Anytime with ❤️ </a></h3>", unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")
    return uploaded_file

def process_uploaded_file(uploaded_file):
    """Processes the uploaded file."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    chain = ConversationalRetrievalChain.from_llm(llm=load_llm(), retriever=db.as_retriever())
    return chain

def initialize_session_state():
    """Initializes the session state."""
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'cache' not in st.session_state:
        st.session_state['cache'] = {}
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def setup_conversational_chat_interface():
    """Sets up the conversational chat interface."""
    with st.container():
        with st.form(key='my_form'):
            user_input = st.text_input("Query:", placeholder="Talk with your csv data here (:", key='input')
            send_button = st.form_submit_button(label='Send')

        if send_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

def display_chat_messages():
    """Displays the chat messages."""
    with st.container():
        for i, generated_message in enumerate(st.session_state['generated']):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(generated_message, key=str(i), avatar_style="thumbs")

def conversational_chat(query):
    """Manages the conversational chat logic, including the cache of responses."""
    if query in st.session_state['cache']:
        return st.session_state['cache'][query]
    
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    st.session_state['cache'][query] = result["answer"]
    return result["answer"]

# Main function to run the application
def main():
    uploaded_file = setup_streamlit_interface()
    if uploaded_file:
        chain = process_uploaded_file(uploaded_file)
        initialize_session_state()
        setup_conversational_chat_interface()
        display_chat_messages()

if __name__ == "__main__":
    main()
