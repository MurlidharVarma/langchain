from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatOpenAI(model_name="gpt-4o", temperature=0.7, max_tokens=500)

result = model.invoke("why 1 paise is called '1 naya paisa' in India?");

print(result.content)