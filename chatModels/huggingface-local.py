from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_length": 512},
)
model = ChatHuggingFace(llm=llm, temperature=0.5)

result = model.invoke("What AI subjects should I study that would become valuable in the next 5 years? Please provide a list of 10 subjects and explain why each is important.")

print(result.content)
