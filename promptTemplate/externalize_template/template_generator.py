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

template.save("template.json")