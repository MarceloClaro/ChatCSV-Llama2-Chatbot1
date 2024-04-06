# Importações de bibliotecas necessárias
import streamlit as st          # Para construir aplicativos web interativos
from streamlit_chat import message  # Para criar uma interface de chat no Streamlit
import pandas as pd           # Para manipulação de dados, especialmente arquivos CSV
import requests               # Para fazer requisições HTTP, como baixar arquivos da internet
import os                     # Para interagir com o sistema de arquivos do sistema operacional

# Caminho onde será armazenado o banco de dados do vetor FAISS
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Variáveis globais: modelo de linguagem (llm) e cadeia de recuperação (chain), além de um prompt inicial
llm = None
chain = None
prompt_initial = "Olá! Pergunte-me qualquer coisa sobre seus dados CSV 🤗"

# Classe para criar e gerenciar um template de prompt
class PromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template              # O template em si, uma string com espaços reservados
        self.input_variables = input_variables  # Variáveis a serem inseridas no template

    # Método para renderizar o template com as variáveis específicas
    def render(self, **kwargs):
        return self.template.format(**kwargs)

# Definição do template inicial de prompt
prompt_template_inicial = """
Não tente inventar uma resposta, se você não sabe, apenas diga que não sabe.
Responda na mesma língua em que a pergunta foi feita.
Use apenas os seguintes pedaços de contexto para responder à pergunta no final.

{contexto}

Pergunta: {questao}
Resposta:"""

# Inicializando a classe PromptTemplate com o template definido anteriormente
PROMPT = PromptTemplate(
    template=prompt_template_inicial,
    input_variables=["contexto", "questao"]
)

# Função para carregar o modelo de linguagem de um determinado caminho
def load_llm(model_path, temperatura, max_tokens):
    if os.path.exists(model_path):
        return CTransformers(
            model=model_path,
            model_type="llama",
            max_new_tokens=max_tokens,
            temperatura=temperatura
        )
    else:
        st.error("Modelo não encontrado: " + model_path)
        return None

# Função para converter dados de um DataFrame (CSV) em texto
def convert_csv_to_text(data):
    text_data = []
    for _, row in data.iterrows():
        text_row = ' '.join(str(val) for val in row)
        text_data.append(text_row)
    return text_data

# Função para processar um arquivo CSV carregado, converter em texto, criar um banco de dados com FAISS e iniciar a cadeia de recuperação
def process_uploaded_file(uploaded_file):
    csv_data = pd.read_csv(uploaded_file)
    text_data = convert_csv_to_text(csv_data)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(text_data, embeddings)
    db.save_local(DB_FAISS_PATH)
    retriever = VectorStoreRetriever(db)
    chain = ConversationalRetrievalChain.from_llm(llm=load_llm(), retriever=retriever)
    return chain

# Função para configurar a página de configurações no Streamlit
def setup_config_page():
    st.title("Configurações do Chatbot")
    
    # Seção para upload de um modelo de linguagem personalizado
    st.subheader("Upload de Modelo de Linguagem Personalizado:")
    uploaded_model = st.file_uploader("Escolha um arquivo de modelo", type=['bin'])

    global llm
    if uploaded_model is not None:
        model_path = os.path.join("uploaded_models", uploaded_model.name)
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("Modelo carregado: " + uploaded_model.name)

        temperatura = st.slider("Temperatura", 0.1, 1.0, 0.5)
        max_tokens = st.slider("Máximo de Tokens Novos", 10, 512, 512)
        llm = load_llm(model_path, temperatura, max_tokens)

    # Seção para editar o template de prompt
    st.subheader("Edite o Template do Prompt:")
    prompt_template = st.text_area("Template do Prompt", value=prompt_template_inicial)
    st.session_state['prompt_template'] = prompt_template

    # Seção para baixar dados CSV
    st.subheader("Baixar Dados CSV:")
    csv_url = st.text_input("URL do Arquivo CSV:")
    download_button = st.button("Baixar CSV")

    # Baixar o arquivo CSV da URL fornecida e processá-lo
    if download_button and csv_url:
        csv_file = download_csv_data(csv_url)
        if csv_file and validate_csv_file(csv_file):
            st.success(f"Arquivo {csv_file} baixado com sucesso!")
            st.session_state['chain'] = process_uploaded_file(csv_file)

# Função para configurar a página de chat no Streamlit
def setup_chat_page():
    st.title("🦙 Chat Inteligente com Dados CSV")
    # Verifica se uma cadeia de recuperação está carregada e permite o chat com o bot
    if 'chain' in st.session_state and st.session_state['chain']:
        with st.container():
            user_input = st.text_input("Pergunta:", placeholder="Converse aqui com os seus dados CSV:")
            send_button = st.button('Enviar')

            # Lógica para enviar perguntas e exibir respostas
            if send_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

            # Exibir as mensagens do chat
            for i, generated_message in enumerate(st.session_state['generated']):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(generated_message, key=str(i))
    else:
        st.error("Nenhum modelo de chat carregado. Por favor, carregue seus dados CSV na página de configurações.")

# Lógica principal de conversação do chatbot
def conversational_chat(questao):
    global chain
    # Buscar na cache primeiro, para evitar processamento repetitivo
    if questao in st.session_state['cache']:
        return st.session_state['cache'][questao]

    contexto = "Seu contexto aqui"
    prompt = PromptTemplate(template=st.session_state.get('prompt_template', prompt_template_inicial), input_variables=["contexto", "questao"])
    formatted_prompt = prompt.render(contexto=contexto, questao=questao)

    # Executa a cadeia de recuperação e gera a resposta
    result = chain({"question": formatted_prompt, "chat_history": st.session_state['history']})
    st.session_state['history'].append((questao, result["answer"]))
    st.session_state['cache'][questao] = result["answer"]
    return result["answer"]

# Função para baixar dados CSV de uma URL fornecida
def download_csv_data(csv_url):
    try:
        r = requests.get(csv_url, stream=True)
        # Baixa o arquivo se a requisição for bem-sucedida
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

# Função para validar se o arquivo CSV pode ser lido corretamente
def validate_csv_file(file_path):
    try:
        pd.read_csv(file_path)
        return True
    except Exception as e:
        st.error(f"Erro ao processar o arquivo CSV: {e}")
        return False

# Função principal para executar o aplicativo Streamlit
def main():
    st.sidebar.title("Navegação")
    # Navegação entre as diferentes páginas do app
    app_mode = st.sidebar.radio("Escolha a página:", ["Configurações", "Chat"])
    
    # Lógica para carregar a página de configurações ou de chat
    if app_mode == "Configurações":
        setup_config_page()
        uploaded_file = st.file_uploader("Envie seus Dados CSV", type="csv")
        if uploaded_file is not None:
            st.session_state['chain'] = process_uploaded_file(uploaded_file)
            st.success("Dados CSV processados e chatbot carregado!")
    elif app_mode == "Chat":
        setup_chat_page()

# Verifica se o script está sendo executado como programa principal e inicializa o app Streamlit
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
