from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

result = embedding.embed_query("What are the best AI subjects to study in the next 5 years?")

print(result)