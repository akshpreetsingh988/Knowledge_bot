import streamlit as st
import pandas as pd
import openai
import os 
from neo4j import GraphDatabase
import yaml
import matplotlib.pyplot as plt
from helper_function import get_description_list, get_description_llm, get_modeling_llm, discovery_component, get_llm
from neo4j_runway import Discovery, GraphDataModeler, PyIngest
from IPython.display import display
from neo4j_runway.code_generation import PyIngestConfigGenerator
from langchain_community.graphs import Neo4jGraph
import textwrap
from prompts import CHAT_PROMPT
from langchain_core.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain
from pyvis.network import Network
import streamlit.components.v1 as components

# import neo4j_data
# from neo4j_runway import GraphDataModeler, LLM

#setting environment variable API key
os.environ['OPENAI_API_KEY'] = '<API_KEY>'

# Define a function to load the OpenAI API key
def load_openai_key():
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        openai.api_key = api_key
        st.success("API key loaded successfully!")
    else:
        st.warning("Please enter your OpenAI API key.")
    return api_key

# Load the OpenAI API key
api_key = load_openai_key()

# Check if API key is provided
if api_key:
    # Step 2: CSV Ingestion Module
    st.sidebar.title("CSV Ingestion")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None: 
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(df)
        # import ipdb; ipdb.set_trace()

        # Step 3: Description Run using LLM
        # st.write("Dataframe Description Report : ")
        # import ipdb; ipdb.set_trace()
        # st.markdown(display(disc.run()))      
        
        

        
        st.title("LLM Discovery Process")
        # if st.button("Run Discovery"):
            # discovery_component(disc)
        DESCRIPTION_DICT = get_description_list(df, api_key)
        disc_llm = get_description_llm() 
        global disc
        disc = Discovery(llm=disc_llm, user_input=DESCRIPTION_DICT, data=df)
        st.write(disc.run())
        st.write(disc.discovery)
        
    
        # import ipdb ; ipdb.set_trace()
        # if disc: 
        st.title("Data Model Visualization")
        # if st.button("Generate Initial Data Model"):
        #     initial_model_component()
        modeling_llm = get_modeling_llm() 
        gdm = GraphDataModeler(llm=modeling_llm, discovery=disc)
        gdm.create_initial_model()
        
        st.graphviz_chart(
                    gdm.current_model.visualize(),
                    use_container_width=True
                )
        
        feedback = st.radio("Did you have more inputs for the current visualization", ("Yes", "No"))
        if feedback == "Yes": 
            suggestions = st.text_area("Enter what you would like : ", height=300) 
            if suggestions: 
                gdm.iterate_model(user_corrections=suggestions)
                gdm.current_model.visualize()
                st.graphviz_chart(
                        gdm.current_model.visualize(),
                        use_container_width=True
                )
        
        st.subheader("Enter the neo4j credentials")
        with st.form("neo4j_credentials_form"):
            username = st.text_input("Neo4j Username", value="neo4j")  # Default value can be modified
            password = st.text_input("Neo4j Password", type="password", value="StrongPassword123")  # Hide password input
            uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
            database = st.text_input("Database Name", value="neo4j")
            csv_name = st.text_input("CSV Filename", value="test_dis.csv")
            
            # Submit button for the form
            submitted = st.form_submit_button("Generate Ingestion Code")
        
        if submitted:
            # Once the form is submitted, instantiate the PyIngestConfigGenerator
            gen = PyIngestConfigGenerator(
                data_model=gdm.current_model,  # Use the current model from gdm
                username=username,
                password=password,
                uri=uri,
                database=database,
                csv_name=csv_name
            )
            
            # Generate the ingestion YAML
            pyingest_yaml = gen.generate_config_string()
            
            # Display the generated YAML in the Streamlit app
            st.subheader("Generated Ingestion YAML")
            st.code(pyingest_yaml, language="yaml")
        
            #Use generated PyIngest yaml config to ingest our CSV into our Neo4j instance
            PyIngest(config=pyingest_yaml, dataframe=df)
            gen.generate_config_yaml(file_name="diseases.yaml")
            
            
        driver = GraphDatabase.driver(uri, auth=(username , password)) 
        
        def fetch_graph_data(): 
            query = """
            MATCH (n)-[r]->(m)
            RETURN n, r, m
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
                # Removing duplicates
                nodes = {n.id: n for n in nodes}.values()
                return nodes, edges
        def close_driver():
            driver.close()
        import ipdb; ipdb.set_trace()
        st.title("Neo4j Graph Visualization")
        nodes, edges = fetch_graph_data()

        net = Network(height="750px", width="100%", notebook=True)

        # Convert frozenset to a string or list
        for node in nodes:
            label = str(node.labels) if isinstance(node.labels, frozenset) else node.labels
            net.add_node(node.id, label=str(node.id), title=label)  # Ensure everything is JSON serializable

        for edge in edges:
            net.add_edge(edge[0], edge[1], title=edge[2])

        # Display the network
        net.show("graph.html")
        HtmlFile = open("graph.html", "r", encoding="utf-8")
        source_code = HtmlFile.read()
        components.html(source_code, height=800, width=1000)

        # Close the Neo4j driver when done
        close_driver()        
        
        # ----
        kg = Neo4jGraph(
            url=uri, username=username, password=password, database=database
        )
        kg.refresh_schema()
        st.divider() 
        st.text(textwrap.fill(kg.schema, 60))
        schema=kg.schema
        
        chat_llm = get_llm("gpt3.5", api_key)
        cypher_prompt = PromptTemplate(
            input_variables=["schema","question"], 
            template=CHAT_PROMPT
        )
        
        cypherChain = GraphCypherQAChain.from_llm(
            chat_llm,
            graph=kg,
            verbose=True,
            cypher_prompt=cypher_prompt,
            top_k=10 # this can be adjusted also
        )
        
        st.header("Please Enter Your Query")
        query = st.text_area("Question", height=300) 
        if query : 
            response = cypherChain.run(query) 
            st.text(response ) 
        
