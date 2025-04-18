import uvicorn
import os 
import sys
from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel
from query_model import QueryModel
from rag.Rag_model import get_final_response
from rag.Rag_model import QueryResponse
from dotenv import load_dotenv

load_dotenv(dotenv_path="/var/task/.env")  # Chemin où le fichier .env est copié

app = FastAPI(debug=True)
handler = Mangum(app)  # Entry point for AWS Lambda.


class SubmitQueryRequest(BaseModel):
    query_text: str


@app.get("/get_query")
def get_query_endpoint(query_id: str) -> QueryModel:
    query = QueryModel.get_item(query_id)
    return query

@app.get("/")
def index():
    return {"Hello": "World"}


#@app.post("/submit_query")
#def submit_query_endpoint(request: SubmitQueryRequest) -> QueryResponse:
#    query_response = get_final_response(request.query_text)
#    return query_response


@app.post("/submit_query")
def submit_query_endpoint(request: SubmitQueryRequest) -> QueryModel:
    # Create the query item, and put it into the data-base.
    new_query = QueryModel(query_text=request.query_text)

   
    # Make a synchronous call to the worker (the RAG/AI app).
    query_response = get_final_response(request.query_text)
    new_query.answer_text = query_response.response_text
    new_query.sources = query_response.sources
    new_query.is_complete = True
    new_query.put_item()

    return new_query

if __name__ == "__main__":
    # Run this as a server directly.
    port = 8000
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run("app_api_handler:app", host="0.0.0.0", port=port)

