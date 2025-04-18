import uvicorn
import os 
import sys
from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel
from rag.Rag_model import get_final_response
from rag.Rag_model import QueryResponse
from dotenv import load_dotenv

load_dotenv(dotenv_path="/var/task/.env")  # Chemin où le fichier .env est copié

app = FastAPI(debug=True)
handler = Mangum(app)  # Entry point for AWS Lambda.


class SubmitQueryRequest(BaseModel):
    query_text: str


@app.get("/")
def index():
    return {"Hello": "World"}


@app.post("/submit_query")
def submit_query_endpoint(request: SubmitQueryRequest) -> QueryResponse:
    query_response = get_final_response(request.query_text)
    return query_response


if __name__ == "__main__":
    # Run this as a server directly.
    port = 8000
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run("app_api_handler:app", host="0.0.0.0", port=port)