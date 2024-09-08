import streamlit as st
import pandas as pd
import openai
import os
from neo4j import GraphDatabase
import matplotlib.pyplot as plt
from helper_function import get_description_list, get_description_llm, get_modeling_llm, get_llm
from neo4j_runway import Discovery, GraphDataModeler, PyIngest
from neo4j_runway.code_generation import PyIngestConfigGenerator
from langchain_community.graphs import Neo4jGraph
import textwrap
from prompts import CHAT_PROMPT
from langchain_core.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain
from pyvis.network import Network
import streamlit.components.v1 as components

#setting environment variable API key
os.environ['OPENAI_API_KEY'] = 'sk-ECHkApmS9AOrsI_QlsZZBaJZyz3qqe8K2L1LBSv56aT3BlbkFJozOLtPVWLq9TzVHQPAJXi5qMV3wLE1NKxPm6AiJGgA'

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'df' not in st.session_state:
    st.session_state.df = None
if 'discovery' not in st.session_state:
    st.session_state.discovery = None
if 'gdm' not in st.session_state:
    st.session_state.gdm = None
if 'neo4j_credentials' not in st.session_state:
    st.session_state.neo4j_credentials = None

# Step 1: Load OpenAI API Key
def load_openai_key():
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        openai.api_key = api_key
        st.success("API key loaded successfully!")
    return api_key

api_key = load_openai_key()

# Step 2: CSV Ingestion
if api_key and st.session_state.step >= 1:
    st.sidebar.title("CSV Ingestion")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.write("### Uploaded Data")
        st.dataframe(df)
        if st.button("Proceed to LLM Discovery"):
            st.session_state.step = 2

# Step 3: LLM Discovery
if api_key and st.session_state.step >= 2 and st.session_state.df is not None:
    st.title("LLM Discovery Process")
    if st.session_state.discovery is None:
        DESCRIPTION_DICT = get_description_list(st.session_state.df, api_key)
        disc_llm = get_description_llm()
        disc = Discovery(llm=disc_llm, user_input=DESCRIPTION_DICT, data=st.session_state.df)
        st.session_state.discovery = disc

    st.write(st.session_state.discovery.run())
    st.write(st.session_state.discovery.discovery)

    if st.button("Proceed to Data Model Visualization"):
        st.session_state.step = 3

# Step 4: Data Model Visualization
if st.session_state.step >= 3 and st.session_state.discovery:
    st.title("Data Model Visualization")

    if st.session_state.gdm is None:
        modeling_llm = get_modeling_llm()
        gdm = GraphDataModeler(llm=modeling_llm, discovery=st.session_state.discovery)
        gdm.create_initial_model()
        st.session_state.gdm = gdm

    st.graphviz_chart(st.session_state.gdm.current_model.visualize(), use_container_width=True)

    feedback = st.radio("Do you have more inputs for the current visualization?", ("Yes", "No"))
    if feedback == "Yes":
        suggestions = st.text_area("Enter your suggestions:")
        if suggestions:
            st.session_state.gdm.iterate_model(user_corrections=suggestions)
            st.graphviz_chart(st.session_state.gdm.current_model.visualize(), use_container_width=True)

    if st.button("Proceed to Neo4j Ingestion"):
        st.session_state.step = 4

# Step 5: Neo4j Ingestion
if st.session_state.step >= 4:
    st.subheader("Enter the Neo4j credentials")
    with st.form("neo4j_credentials_form"):
        username = st.text_input("Neo4j Username", value="neo4j")
        password = st.text_input("Neo4j Password", type="password", value="StrongPassword123")
        uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
        database = st.text_input("Database Name", value="neo4j")
        csv_name = st.text_input("CSV Filename", value="test_dis.csv")
        submitted = st.form_submit_button("Generate Ingestion Code")

    if submitted:
        gen = PyIngestConfigGenerator(
            data_model=st.session_state.gdm.current_model,
            username=username,
            password=password,
            uri=uri,
            database=database,
            csv_name=csv_name
        )
        pyingest_yaml = gen.generate_config_string()
        st.subheader("Generated Ingestion YAML")
        st.code(pyingest_yaml, language="yaml")
        PyIngest(config=pyingest_yaml, dataframe=st.session_state.df)
        gen.generate_config_yaml(file_name="diseases.yaml")

        st.session_state.neo4j_credentials = {
            'username': username,
            'password': password,
            'uri': uri,
            'database': database
        }
        st.session_state.step = 5

# Step 6: Graph Visualization with PyVis (Optimized)
if st.session_state.step >= 5 and st.session_state.neo4j_credentials:
    driver = GraphDatabase.driver(
        st.session_state.neo4j_credentials['uri'],
        auth=(st.session_state.neo4j_credentials['username'], st.session_state.neo4j_credentials['password'])
    )

    # Fetch graph data but limit nodes/edges
    def fetch_graph_data(limit=100):
        query = f"""
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT {limit}
        """
        with driver.session() as session:
            results = session.run(query)
            nodes = []
            edges = []
            for record in results:
                n = record["n"]
                m = record["m"]
                r = record["r"]
                nodes.append(n)
                nodes.append(m)
                edges.append((n.id, m.id, r.type))
            nodes = {n.id: n for n in nodes}.values()  # Remove duplicates
            return nodes, edges

    # Default limit of nodes/edges to load
    if 'graph_limit' not in st.session_state:
        st.session_state.graph_limit = 100

    st.title("Neo4j Graph Visualization")

    if st.button("Load More Nodes"):
        st.session_state.graph_limit += 100  # Increment limit by 100 nodes/edges

    nodes, edges = fetch_graph_data(limit=st.session_state.graph_limit)

    # Visualize graph with PyVis
    net = Network(height="750px", width="100%", notebook=True)

    for node in nodes:
        label = str(node.labels) if isinstance(node.labels, frozenset) else node.labels
        net.add_node(node.id, label=str(node.id), title=label)

    for edge in edges:
        net.add_edge(edge[0], edge[1], title=edge[2])

    net.show("graph.html")
    HtmlFile = open("graph.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=800, width=1000)

    driver.close()

# Step 7: CypherChain LLM Interaction
if st.session_state.step >= 5 and st.session_state.neo4j_credentials:
    st.header("Please Enter Your Query")
    query = st.text_area("Question",height=500)
    if query:
        kg = Neo4jGraph(
            url=st.session_state.neo4j_credentials['uri'],
            username=st.session_state.neo4j_credentials['username'],
            password=st.session_state.neo4j_credentials['password'],
            database=st.session_state.neo4j_credentials['database']
        )
        kg.refresh_schema()
        st.divider()
        st.text(textwrap.fill(kg.schema, 60))

        chat_llm = get_llm("gpt3.5", api_key)
        cypher_prompt = PromptTemplate(input_variables=["schema", "question"], template=CHAT_PROMPT)
        cypherChain = GraphCypherQAChain.from_llm(
            chat_llm, graph=kg, verbose=True, cypher_prompt=cypher_prompt, top_k=10
        )
        response = cypherChain.run(query)
        st.text(response)
