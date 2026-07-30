import os
import requests
from fastapi import FastAPI, Request
from agent import run_agent

app = FastAPI()

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    
    if "message" in data:
        message = data["message"]
        chat_id = message["chat"]["id"]
        text = message.get("text", "")
        
        public_log_url = f"{RENDER_EXTERNAL_URL}/run.jsonl"
        
        # Call your stateless agent
        agent_reply_json = run_agent(text, public_log_url)
        
        # Send response back to Telegram
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": agent_reply_json
        }
        requests.post(url, json=payload)
        
    return {"status": "ok"}

@app.get("/run.jsonl")
async def get_log():
    from fastapi.responses import FileResponse
    if os.path.exists("run.jsonl"):
        return FileResponse("run.jsonl", media_type="application/json")
    return {"error": "No logs yet"}

# Optional: root check to see if service is alive
@app.get("/")
async def root():
    return {"status": "Render service is running"}
