from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# to get the embedding for a single query
result = embedding.embed_query("What are the best AI subjects to study in the next 5 years?")
print(result)   

#to get the embedding for a list of documents
document = [
    "What are the best AI subjects to study in the next 5 years?",
    "Quick brown fox jumps over the lazy dog.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
]
result = embedding.embed_documents(document)
print(result)
