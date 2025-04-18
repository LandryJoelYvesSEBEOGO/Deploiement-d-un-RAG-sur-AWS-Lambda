import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings
from typing import List
from config.Config import Config
import shutil

load_dotenv()

IS_USING_IMAGE_RUNTIME = bool(os.environ.get("IS_USING_IMAGE_RUNTIME", False))
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

# Liste des URLs à scraper
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://arxiv.org/pdf/2503.11651v1"
]

# Répertoire pour ChromaDB
CHROMA_DB_DIR = os.path.join("data", "chroma_db")

def load_web_documents(urls: List[str]):
    """Charge les documents à partir d'URLs."""
    docs = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in docs for item in sublist]

def load_pdf_documents(pdf_paths: List[str]):
    """Charge les documents à partir de fichiers PDF."""
    docs = [PyPDFLoader(path).load() for path in pdf_paths]
    return [item for sublist in docs for item in sublist]

def split_documents(docs_list: List, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Divise les documents en morceaux pour un traitement plus efficace."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs_list)

def get_runtime_chroma_path():
    return f"/tmp/{CHROMA_DB_DIR}" if IS_USING_IMAGE_RUNTIME else CHROMA_DB_DIR

def copy_chroma_to_tmp():
    dst_chroma_path = get_runtime_chroma_path()

    if not os.path.exists(dst_chroma_path):
        os.makedirs(dst_chroma_path)

    tmp_contents = os.listdir(dst_chroma_path)
    if len(tmp_contents) == 0:
        print(f"Copying ChromaDB from {CHROMA_DB_DIR} to {dst_chroma_path}")
        os.makedirs(dst_chroma_path, exist_ok=True)
        shutil.copytree(CHROMA_DB_DIR, dst_chroma_path, dirs_exist_ok=True)
    else:
        print(f"✅ ChromaDB already exists in {dst_chroma_path}")

def get_or_create_chroma_db(doc_splits: List, persist_directory: str = None, clear_db: bool = False):
    """
    Crée ou recharge une base de données Chroma.
    """
    if persist_directory is None:
        persist_directory = get_runtime_chroma_path()

    print(f"[DEBUG] persist_directory used for ChromaDB: {persist_directory}")

    if IS_USING_IMAGE_RUNTIME:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
        copy_chroma_to_tmp()

    embedding_model = NomicEmbeddings(
        model=Config.NomicEmbeddings_model,
    )

    if clear_db:
        print(f"[INFO] Suppression complète de ChromaDB dans {persist_directory}...")
        shutil.rmtree(persist_directory, ignore_errors=True)
        print("[INFO] ChromaDB supprimée. Recréation en cours...")
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
    else:
        print("[INFO] Chargement ou création de la base ChromaDB...")
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory, exist_ok=True)
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            embedding=embedding_model,
            persist_directory=persist_directory
        )

    return vectorstore

def load_pdf_documents_from_folder(folder_path: str):
    """Charge tous les fichiers PDF présents dans un dossier donné."""
    pdf_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".pdf")
    ]
    return load_pdf_documents(pdf_paths)

def get_retriever(k: int = 3):
    """Crée un retriever pour récupérer les documents les plus pertinents."""

    # Charger les documents web
    web_docs = load_web_documents(urls)

    # Charger automatiquement tous les PDFs du dossier "Dataset"
    pdf_docs = load_pdf_documents_from_folder(os.path.join("data", "raw"))

    # Fusionner les deux listes de documents
    all_docs = web_docs + pdf_docs

    # Diviser les documents en segments
    doc_splits = split_documents(all_docs)

    # Utiliser le bon chemin (selon l'environnement)
    vectorstore = get_or_create_chroma_db(doc_splits, clear_db=False)

    return vectorstore.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    retriever = get_retriever()
    # Exemple d’utilisation
    # print(retriever.search("What is a transformer?"))
