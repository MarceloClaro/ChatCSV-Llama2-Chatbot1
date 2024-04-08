import streamlit as st
import os
import openai
import time
import tempfile

os.environ['OPENAI_API_KEY'] = "YOUR_OPEN_AI_API_KEY"

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    return tmp_file_path

def run_query(query, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": query}],
        max_tokens=150
    )
    return response.choices[0].message['content']

def main():
    st.title("CSV Agent Chat with GPT-3.5")
    st.markdown("<h3 style='text-align: center; color: white;'></a></h3>", unsafe_allow_html=True)

    st.sidebar.write("Faça upload do arquivo .gguf:")
    uploaded_file = st.sidebar.file_uploader("Carregar modelo .gguf", type="gguf", accept_multiple_files=False, key='gguf', help="O arquivo deve ser superior a 1 GB")

    if uploaded_file:
        gguf_path = process_uploaded_file(uploaded_file)
        st.write("Arquivo .gguf carregado com sucesso!")
        
        print("\nCSV Agent is ready to assist you!")

        while True:
            query = st.text_input("\nO que você gostaria de saber?: ")
            if query.lower() == "exit":
                break
            if query.strip() == "":
                continue
            
            start = time.time()
            answer = run_query(query)
            end = time.time()
            
            st.write(answer)
            st.write(f"\n> Resposta (demorou {round(end - start, 2)} s.)")

if __name__ == "__main__":
    main()
