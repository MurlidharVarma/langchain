from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct:novita",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm, temperature=0.5)

result = model.invoke("What AI subjects should I study that would become valuable in the next 5 years? Please provide a list of 10 subjects and explain why each is important.");

print(result.content)