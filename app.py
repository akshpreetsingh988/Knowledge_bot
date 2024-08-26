import streamlit as st
import pandas as pd
from py2neo import Graph, Node, Relationship
import networkx as nx
import matplotlib.pyplot as plt

# Initialize Neo4j connection
def init_neo4j_connection(uri, user, password):
    graph = Graph(uri, auth=(user, password))
    return graph

# Load CSV or Excel file
def load_file(file):
    if file.name.endswith('csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('xls', 'xlsx')):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file type")
        return None
    return df

# Display the CSV as a graph
def display_graph(df, graph):
    G = nx.Graph()

    # Example: Create nodes and relationships
    for index, row in df.iterrows():
        node1 = Node("Entity", name=row['Column1'])
        node2 = Node("Entity", name=row['Column2'])
        rel = Relationship(node1, "RELATES_TO", node2)
        graph.merge(node1, "Entity", "name")
        graph.merge(node2, "Entity", "name")
        graph.merge(rel)
        G.add_edge(row['Column1'], row['Column2'])

    # Visualize using NetworkX
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1500, edge_color='black', linewidths=1, font_size=15)
    plt.show()
    st.pyplot(plt)

# Streamlit UI
def main():
    st.sidebar.title("Upload File")
    uploaded_file = st.sidebar.file_uploader("Choose a file")

    if uploaded_file is not None:
        df = load_file(uploaded_file)
        if df is not None:
            st.write("DataFrame:")
            st.write(df)

            st.sidebar.title("Neo4j Connection")
            uri = st.sidebar.text_input("Bolt URI", "bolt://localhost:7687")
            user = st.sidebar.text_input("User", "neo4j")
            password = st.sidebar.text_input("Password", type="password")

            if st.sidebar.button("Connect and Display Graph"):
                graph = init_neo4j_connection(uri, user, password)
                display_graph(df, graph)

if __name__ == "__main__":
    main()
