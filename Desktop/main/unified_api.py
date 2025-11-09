import os
import subprocess
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uvicorn

app = FastAPI(title="Unified AI/ML/DL API Server")

# Define your model endpoints
REPOS = {
    "ml_deploy": "ML-Projects-with-Deployment",
    "torchserve": "torchserve-dashboard",
    "langchain": "langchain",
    "haystack": "haystack",
    "madewithml": "MadeWithML",
    "ml_template": "ml-serving-template",
    "aiforbeginners": "AI-For-Beginners",
    "deepspeed": "DeepSpeed",
    "sd_webui": "stable-diffusion-webui"
}

def make_request_with_retry(url, method='GET', max_retries=5, **kwargs):
    """
    Make an HTTP request with retry logic for handling 429 (Too Many Requests) and 401 (Invalid API Key) errors.
    Uses linear backoff for retries. Does not retry on 401.
    """
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code == 429:
                wait_time = 5 * (attempt + 1)  # Linear backoff
                print(f"Rate limited. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            elif response.status_code == 401:
                # Invalid API key or unauthorized, do not retry
                print(f"Error: Invalid API key or unauthorized access (401). Check your API key at https://openrouter.ai/keys.")
                return response
            return response
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = 5 * (attempt + 1)
            print(f"Request failed. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    return None

class Request(BaseModel):
    input: str
    repo: str

class OpenRouterRequest(BaseModel):
    input: str
    model: str = "custom/blackbox-base"

@app.post("/run")
def run_model(req: Request):
    repo_path = REPOS.get(req.repo)
    if not repo_path:
        return {"error": "Invalid repo name"}
    
    # Example: Run a script from the repo
    try:
        result = subprocess.run(
            ["python", "main.py"],
            cwd=os.path.join(os.getcwd(), repo_path),
            capture_output=True,
            text=True
        )
        return {
            "timestamp": datetime.now().isoformat(),
            "output": result.stdout,
            "error": result.stderr
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/update/{repo}")
def update_repo(repo: str):
    repo_path = REPOS.get(repo)
    if not repo_path:
        return {"error": "Invalid repo name"}
    
    try:
        result = subprocess.run(
            ["git", "pull"],
            cwd=os.path.join(os.getcwd(), repo_path),
            capture_output=True,
            text=True
        )
        return {
            "timestamp": datetime.now().isoformat(),
            "output": result.stdout,
            "error": result.stderr
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/timeline")
def timeline():
    return {
        "timestamp": datetime.now().isoformat(),
        "repos": list(REPOS.keys())
    }

@app.post("/ollama")
def call_ollama(req: OpenRouterRequest):
    """
    Call Ollama API with the given input and model.
    Ollama runs locally on http://localhost:11434
    """
    # Check if Ollama is running
    try:
        health_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if health_response.status_code != 200:
            raise HTTPException(status_code=503, detail="Ollama server not available. Make sure Ollama is running on localhost:11434")
    except requests.RequestException:
        raise HTTPException(status_code=503, detail="Cannot connect to Ollama server. Make sure Ollama is running on localhost:11434")

    # For local testing without real model, use mock response
    if req.model == "mock":
        # Simulate a successful response
        return {
            "model": "mock-model",
            "created_at": datetime.now().isoformat(),
            "response": f"Mock response to input: {req.input}",
            "done": True
        }

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": req.model,
        "prompt": req.input,
        "stream": False
    }

    response = make_request_with_retry(url, method='POST', json=payload)
    if response is None:
        raise HTTPException(status_code=500, detail="Request failed after retries.")
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
