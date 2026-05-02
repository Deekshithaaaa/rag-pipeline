import openai
import chromadb
import langchain
from dotenv import load_dotenv
import os

load_dotenv()

print("✅ All imports successful")
print(f"✅ OpenAI key loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'NO - CHECK .env'}")
print(f"✅ LangChain version: {langchain.__version__}")
print(f"✅ ChromaDB version: {chromadb.__version__}")