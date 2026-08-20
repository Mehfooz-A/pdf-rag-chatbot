import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load GEMINI_API_KEY from .env, same as before
load_dotenv()

# 1. The model — LangChain's wrapper around Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=os.environ["GEMINI_API_KEY"]
)

# 2. The prompt template — {topic} and {audience} are blanks filled in later
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one sentence for a {audience}."
)

# 3. The output parser — extracts plain text from the model's response
parser = StrOutputParser()

# Chain them together with the pipe operator
chain = prompt | model | parser

# Run the chain, filling in the template's blanks
result = chain.invoke({"topic": "RAG", "audience": "5-year-old"})

print(result)