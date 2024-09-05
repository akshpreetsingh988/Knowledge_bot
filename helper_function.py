import ast 
from langchain_community.chat_models import ChatOpenAI
from prompts import DESCRIPTION_PROMPT
from neo4j_runway.llm.openai import OpenAIDiscoveryLLM, OpenAIDataModelingLLM
import streamlit as st 

def get_llm(model ,  api_key): 
    if model == "gpt3.5": 
        chat_llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-3.5-turbo",  # or use "gpt-4" if you have access
        )
        return chat_llm

def get_description_list(df, api_key): 
    LLM = get_llm("gpt3.5", api_key) 
    
    column_names = ", ".join(df.columns)
    temp_df = df[:3].to_string(index = False) 
    
    prompt = DESCRIPTION_PROMPT.format(COLUMN_NAMES = column_names, df = temp_df)
    response = LLM.invoke(prompt).content
    
    DATA_DESCRIPTION = response.split("=", 1)[1].strip()
    DATA_DESCRIPTION = ast.literal_eval(DATA_DESCRIPTION)
    return DATA_DESCRIPTION

def get_description_llm(): 
    disc_llm = OpenAIDiscoveryLLM()
    return disc_llm

def get_modeling_llm(): 
    modeling_llm = OpenAIDataModelingLLM()
    return modeling_llm


def discovery_component(disc, show: bool = True) -> None:
            """
            Discovery component. Display the LLM discovery step in Streamlit UI.
            """
            
            # Initialize session state if it doesn't exist
            if "discovery_ran" not in st.session_state:
                st.session_state["discovery_ran"] = False
                st.session_state["show_discovery"] = show
                st.session_state["discovery_summary"] = None

            with st.expander("Discovery", expanded=st.session_state["show_discovery"]):
                if not st.session_state["discovery_ran"]:
                    # Run the Discovery object
                    disc.run()
                    st.session_state["discovery_summary"] = disc.discovery  # Store the discovery result in session state

                # Display the summary if available
                if st.session_state["discovery_summary"]:
                    st.write(st.session_state["discovery_summary"])

                # Mark discovery as done
                st.session_state["discovery_ran"] = True
                st.session_state["show_initial_data_model"] = True