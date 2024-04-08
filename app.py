import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Data App')
st.title('🦜🔗 Ask the Data App')

# Load CSV file
def load_csv(input_csv):
    df = pd.read_csv(input_csv)
    with st.expander('See DataFrame'):
        st.write(df)
    return df

# Generate LLM response
def generate_response(csv_file, input_query, openai_api_key):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0.2, openai_api_key=openai_api_key)
    df = load_csv(csv_file)
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
    response = agent.run(input_query)
    return st.success(response)

# Input widgets
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
question_list = [
    'How many rows are there?',
    'What is the range of values for MolWt with logS greater than 0?',
    'How many rows have MolLogP value greater than 0.',
    'Other'
]
query_text = st.selectbox('Select an example query:', question_list, index=0 if uploaded_file else None, disabled=not uploaded_file)
openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))

# App logic
if uploaded_file and query_text != 'Other' and openai_api_key:
    st.header('Output')
    generate_response(uploaded_file, query_text, openai_api_key)

elif uploaded_file and query_text == 'Other':
    custom_query = st.text_input('Enter your query:', placeholder='Enter query here ...')
    if st.button('Submit') and custom_query and openai_api_key:
        st.header('Output')
        generate_response(uploaded_file, custom_query, openai_api_key)

elif not openai_api_key:
    st.warning('Please enter your OpenAI API key!', icon='⚠')

elif not uploaded_file:
    st.warning('Please upload a CSV file!', icon='⚠')
