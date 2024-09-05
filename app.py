import streamlit as st
import pandas as pd
import openai
import os 
from neo4j import GraphDatabase
import yaml
import matplotlib.pyplot as plt
from helper_function import get_description_list, get_description_llm, get_modeling_llm, discovery_component
from neo4j_runway import Discovery, GraphDataModeler, PyIngest
from IPython.display import display
from neo4j_runway.code_generation import PyIngestConfigGenerator
from langchain_community.graphs import Neo4jGraph
import textwrap
from prompts import CHAT_PROMPT
from langchain_core.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain

# from neo4j_runway import GraphDataModeler, LLM

#setting environment variable API key
os.environ['OPENAI_API_KEY'] = 'sk-5tLuQzXGUemKeK54H-jdat1WZs7uFRvYDvWU83p-NZT3BlbkFJ2KPYOXc2YWLMW8bnv30KaxUYJQaAZx8jkPJ-tOMeoA'

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
        if st.button("Run Discovery"):
            # discovery_component(disc)
            DESCRIPTION_DICT = get_description_list(df, api_key)
            disc_llm = get_description_llm() 
            
            disc = Discovery(llm=disc_llm, user_input=DESCRIPTION_DICT, data=df)
            st.write(disc.run())
            st.write(disc.discovery)
        
        # def initial_model_component(show: bool = True) -> None:
        #     """
        #     Display the initial data model JSON and its visualization in Streamlit UI.
        #     """
            
        #     # Initialize session state for the initial model creation if not already done
        #     if "initial_model_created" not in st.session_state:
        #         st.session_state["initial_model_created"] = False
            
        #     if "modeler" not in st.session_state:
        #         st.session_state["modeler"] = None

        #     # Display section for Data Model V1
        #     with st.expander("Data Model V1", expanded=show):
        #         if not st.session_state["initial_model_created"]:
        #             # Initialize GraphDataModeler
        #             modeling_llm = get_modeling_llm() 
        #             st.session_state["modeler"] = GraphDataModeler(
        #                 llm=modeling_llm,
        #                 discovery=disc # Using discovery from session state
        #             )
        #             # Create the initial model
        #             st.session_state["modeler"].create_initial_model()
        #             st.session_state["initial_model_created"] = True

        #         # Display the JSON dump of the initial model
        #         st.json(st.session_state["modeler"].model_history[0].model_dump(), expanded=False)
                
        #         # Display the visualized model using Graphviz
                


        import ipdb ; ipdb.set_trace()
        if disc: 
            st.title("Data Model Visualization")
            if st.button("Generate Initial Data Model"):
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
                suggestions = st.text_input("Enter what you would like : ") 
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
                
                kg = Neo4jGraph(
                    url=uri, username=username, password=password, database=database
                )
                kg.refresh_schema()
                st.divider() 
                st.text(textwrap.fill(kg.schema, 60))
                schema=kg.schema
                
                chat_llm = get_description_llm()
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
                query = st.text_input("Question") 
                response = cypherChain.run(query) 
                st.text(response ) 
                