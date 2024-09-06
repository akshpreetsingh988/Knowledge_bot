import streamlit as st
import pandas as pd
import openai
import os
from neo4j import GraphDatabase
import yaml
import matplotlib.pyplot as plt
from helper_function import get_description_list, get_description_llm, get_modeling_llm, discovery_component
from neo4j_runway import Discovery, GraphDataModeler, PyIngest
from langchain_community.graphs import Neo4jGraph
import textwrap
from prompts import CHAT_PROMPT
from langchain_core.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain

# Use a custom theme or set colors
st.set_page_config(page_title="LLM & Neo4j Data Modeling", page_icon="🔍", layout="wide")

# Add a sidebar title with a subtle background color
st.sidebar.markdown("<h2 style='background-color: #f63366; color: white; padding: 10px;'>Data Ingestion App</h2>", unsafe_allow_html=True)

# Organize main UI in a column layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Step 1: Enter OpenAI API Key")
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        openai.api_key = api_key
        st.success("API key loaded successfully!")
    else:
        st.warning("Please enter your OpenAI API key.")

with col2:
    st.markdown("### Step 2: CSV Ingestion")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("#### Uploaded Data")
        st.dataframe(df)

# Use expanders to declutter the interface
with st.expander("LLM Discovery Process"):
    st.title("LLM Discovery Process")
    if uploaded_file is not None:
        DESCRIPTION_DICT = get_description_list(df, api_key)
        disc_llm = get_description_llm()
        global disc
        disc = Discovery(llm=disc_llm, user_input=DESCRIPTION_DICT, data=df)
        st.write(disc.run())
        st.write(disc.discovery)

# Add an expander for model visualization
with st.expander("Data Model Visualization"):
    st.title("Data Model Visualization")
    if uploaded_file is not None:
        modeling_llm = get_modeling_llm()
        gdm = GraphDataModeler(llm=modeling_llm, discovery=disc)
        gdm.create_initial_model()
        st.graphviz_chart(gdm.current_model.visualize(), use_container_width=True)

        feedback = st.radio("Did you have more inputs for the current visualization", ("Yes", "No"))
        if feedback == "Yes":
            suggestions = st.text_input("Enter what you would like : ")
            if suggestions:
                gdm.iterate_model(user_corrections=suggestions)
                st.graphviz_chart(gdm.current_model.visualize(), use_container_width=True)

# Neo4j credentials and ingestion code generation section
with st.expander("Neo4j Ingestion Setup"):
    st.subheader("Enter Neo4j Credentials")
    with st.form("neo4j_credentials_form"):
        username = st.text_input("Neo4j Username", value="neo4j")
        password = st.text_input("Neo4j Password", type="password", value="StrongPassword123")
        uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
        database = st.text_input("Database Name", value="neo4j")
        csv_name = st.text_input("CSV Filename", value="test_dis.csv")
        submitted = st.form_submit_button("Generate Ingestion Code")

        if submitted:
            gen = PyIngestConfigGenerator(
                data_model=gdm.current_model,
                username=username,
                password=password,
                uri=uri,
                database=database,
                csv_name=csv_name
            )
            pyingest_yaml = gen.generate_config_string()
            st.subheader("Generated Ingestion YAML")
            st.code(pyingest_yaml, language="yaml")

            PyIngest(config=pyingest_yaml, dataframe=df)
            gen.generate_config_yaml(file_name="diseases.yaml")

# Query section
with st.expander("Query the Graph"):
    st.header("Ask a Question")
    query = st.text_input("Enter your question")
    if query:
        response = cypherChain.run(query)
        st.text(response)
