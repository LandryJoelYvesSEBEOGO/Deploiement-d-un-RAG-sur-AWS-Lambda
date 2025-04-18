from dotenv import load_dotenv
import os

# Chargement des variables d'environnement
load_dotenv(dotenv_path="/var/task/.env")  # Chemin où le fichier .env est copié
#load_dotenv(dotenv_path=".env")
# Environment setup
NOMIC_BASE_PATH = "/tmp/.nomic"
os.environ["NOMIC_BASE_PATH"] = NOMIC_BASE_PATH
os.makedirs(NOMIC_BASE_PATH, exist_ok=True)

class Config:

    # Configuration des environnemnt
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"
    
    #Configuration des Keys 
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    Nomic_api_key = os.getenv("NOMIC_API_KEY")

    #Configuration des models 
    GROQ_model="llama3-8b-8192"
    NomicEmbeddings_model="nomic-embed-text-v1.5"


# Global variables
class SPEAKER_TYPES:
  USER = "user"
  BOT = "bot"


#if __name__ == "__main__":
    #print(Config.GROQ_API_KEY)
    #print(Config.GROQ_model)
    #print(Config.NomicEmbeddings_model)
    #print(Config.Nomic_api_key)
    #print(Config.LANGSMITH_API_KEY)
    #print(SPEAKER_TYPES.USER)
    #print(SPEAKER_TYPES.BOT)

    # Data_processing.py

#print("Nom du module :", __name__)
#print("Package du module :", __package__)
