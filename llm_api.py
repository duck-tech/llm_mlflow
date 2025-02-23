from fastapi import FastAPI, Request, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import json

app = FastAPI()

# Ollama 伺服器 URL
OLLAMA_URL = "http://localhost:11434/api/generate"

# 設定有效 API Keys（可擴充成從環境變數或資料庫讀取）
VALID_API_KEYS = {"kelly"}

def verify_api_key(api_key: str = Query(..., description="API Key for authentication")):
    """檢查 API Key 是否有效"""
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

class ChatRequest(BaseModel):
    messages: list[dict]  # 例如：[{ "role": "user", "content": "你好!"}]
    model: str = "llama3.1"  # 預設使用 Llama3.1 7B
    temperature: float = 0.0
    top_p: float = 0.95
    frequency_penalty: float = 0
    presence_penalty: float = 0
    max_tokens: int = 2000
    n: int = 1
    seed: int = 0

@app.post("/chat")
async def chat(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    try:
        # 轉換 messages 格式為 Ollama 需要的 prompt
        prompt = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in request.messages)

        headers = {
            "X_AI_GN_TOKEN": f"Bearer {api_key}",
            "X_UID": "TESTUSE",
            "Content-Type": "application/json",
        }

        payload = {
            "model": request.model,
            "prompt": prompt,
            "stream": True,  # 啟用 Streaming
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "max_tokens": request.max_tokens,
                "seed": request.seed
            }
        }

        # 發送請求到 Ollama，攜帶 Headers
        response = requests.post(OLLAMA_URL, json=payload, headers=headers, stream=True)

        # 逐行讀取 Ollama 回應，並流式傳回給客戶端
        def event_stream():
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"] + " "

        return StreamingResponse(event_stream(), media_type="text/plain")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
