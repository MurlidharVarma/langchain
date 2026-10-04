import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm, temperature=0)


# Prompt Template begins here
from langchain_core.prompts import PromptTemplate

templateString ="""
Create a response in less than 10 words. Ensure the response is concise, crips and clear. Avoid using filler words or unnecessary phrases.

Example:

User Input: What AI subjects should I study that would become valuable in the next 5 years
Response: Study AI subjects like machine learning, NLP, and robotics for future relevance.

User Input: What is the captial of India?
Response: New Delhi

User Input: What is teleporation?
Response: Teleportation is the transfer of matter or energy instantly.

User Input: {user_input}
Response:"""

template = PromptTemplate(
    input_variables=["user_input"],
    template=templateString,
    validate_template=True
)


# Streamlit UI begins here


st.header("AI Prompt Template UI")

user_input = st.text_input("Your Question?")


if st.button("Ask"):
    result=None
    templated_input = None
    templated_input = template.invoke({"user_input": user_input})
    result = model.invoke(templated_input)
    st.write("Response:")
    str = result.content
    str_res = str[str.rfind("Response:")+9:]
    st.write(str_res)

## RUN AS 
## python -m streamlit run simplePromptTemplate_ui.py