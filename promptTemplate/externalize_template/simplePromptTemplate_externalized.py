from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm, temperature=0.5)

#Prompt Template begins here
from langchain_core.prompts import PromptTemplate,load_prompt

template = load_prompt("promptTemplate/externalize_template/template.json")

templated_input = template.invoke({"user_input": "Which AI subjects should I study that would become valuable in the next 5 years?"})

print(templated_input)

result = model.invoke(templated_input)

print(result.content)