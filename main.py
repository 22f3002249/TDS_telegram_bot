import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from agent import run_agent

app = FastAPI()

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
VERCEL_EXTERNAL_URL = os.environ.get("VERCEL_EXTERNAL_URL")

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    
    if "message" in data:
        message = data["message"]
        chat_id = message["chat"]["id"]
        text = message.get("text", "")
        
        public_log_url = f"{VERCEL_EXTERNAL_URL}/run.jsonl"
        
        agent_reply = run_agent(text, public_log_url)
        
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": agent_reply
        }
        requests.post(url, json=payload)
        
    return {"status": "ok"}

@app.get("/run.jsonl")
async def get_log():
    log_path = "/tmp/run.jsonl"
    if os.path.exists(log_path):
        return FileResponse(log_path, media_type="application/json")
    return {"error": "No logs yet"}
