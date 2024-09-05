DESCRIPTION_PROMPT = """"
A table with the following column names : {COLUMN_NAMES} and the the sample dataframe is as follows : {df}. 
###Instruction### 
You are a spreadsheet analyzer who will analyze the dataframe and write a small summary of each of the columns of the dataframe. 
The output will be a dictionary have column names as keys and their description as values . 

Example output of python dictionary : 

DATA_DESCRIPTION = {{
'column1': 'description1',
'column2': 'description2'
}}
"""


CHAT_PROMPT = """
Let's think step by step:

Step1: Task:
Generate an effective and concise Cypher statement with less than 256 characteres to query a graph database
Do not comment the code.

Step 2: Get to know the database schema: {schema}

Step 3: Instructions:
- In the cypher query, ONLY USE the provided relationship types and properties that appear in the schema AND in the user question.
- In the cypher query, do not use any other relationship types or properties in the user's question that are not contained in the provided schema.
- Regarding Age, NEVER work with the age itself. For example: 24 years old, use interval: more than 20 years old.
- USE ONLY ONE statement for Age, always use 'greater than', never 'less than' or 'equal'.
- DO NOT USE property keys that are not in the database.

Step 4: Examples: 
Here are a few examples of generated Cypher statements for particular questions:

4.1 Which diseases present high blood pressure?
MATCH (d:Disease)
MATCH (d)-[r:HAS_LAB]->(l)
WHERE l.bloodPressure = 'High'
RETURN d.name

4.2 Which diseases present indicators as high blood pressure?
// Match the Disease nodes
MATCH (d:Disease)
// Match HAS_LAB relationships from Disease nodes to Lab nodes
MATCH (d)-[r:HAS_LAB]->(l)
MATCH (d)-[r2:HAS_OUTCOME]->(o)
// Ensure the Lab nodes have the bloodPressure property set to 'High'
WHERE l.bloodPressure = 'High' AND o.result='Positive'
RETURN d, properties(d) AS disease_properties, r, properties(r) AS relationship_properties, l, properties(l) AS lab_properties

4.3 What is the name of a disease of the elderly where the patient presents high blood pressure, high cholesterol, fever, fatigue
MATCH (d:Disease)
MATCH (d)-[r1:HAS_LAB]->(lab)
MATCH (d)-[r2:HAS_SYMPTOM]->(symptom)
MATCH (symptom)-[r3:HAS_DEMOGRAPHICS]->(demo)
WHERE lab.bloodPressure = 'High' AND lab.cholesterolLevel = 'High' AND symptom.fever = 'Yes' AND symptom.fatigue = 'Yes' AND TOINTEGER(demo.age) >40
RETURN d.name

4.4 What disease gives you fever, fatigue, no cough, no short breathe in people with high cholesterol?
MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
WHERE s.fever = 'Yes' AND s.fatigue = 'Yes' AND s.difficultyBreathing = 'No' AND s.cough = 'No'
MATCH (d:Disease)-[r1:HAS_LAB]->(lab:HealthIndicator)
MATCH (d)-[r2:HAS_OUTCOME]->(o:Outcome)
WHERE lab.cholesterolLevel='High' AND o.result='Positive'
RETURN d, properties(d) AS disease_properties, r, properties(r) AS relationship_properties


Step 5. These are the values allowed for each entity:
- Fever: Indicates whether the patient has a fever (Yes/No).
- Cough: Indicates whether the patient has a cough (Yes/No).
- Fatigue: Indicates whether the patient experiences fatigue (Yes/No).
- Difficulty Breathing': 'Indicates whether the patient has difficulty breathing (Yes/No).
- Age: The age of the patient in years.
- Gender: The gender of the patient (Male/Female).
- Blood Pressure: The blood pressure level of the patient (Normal/High).
- Cholesterol Level: The cholesterol level of the patient (Normal/High).
- Outcome Variable: The outcome variable indicating the result of the diagnosis or assessment for the specific disease (Positive/Negative).

Step 6. Answer the question {question}."""