import os
import json
import requests

GEMINI_MODELS = [
    os.environ.get("PRIMARY_MODEL", "gemini-3.6-flash"),       # Primary Model
    os.environ.get("FALLBACK_MODEL_1", "gemini-3.5-flash-lite"),       # Fallback 1
    os.environ.get("FALLBACK_MODEL_2", "gemini-3.1-flash-lite"),     # Fallback 2
]

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 

LOG_FILE = "/tmp/run.jsonl"

def log_step(step_data: dict):
    """Append a step to the local temporary JSONL log."""
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(step_data) + "\n")

def run_agent(question: str, public_log_url: str) -> str:
    # Clear or initialize log for this run
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    log_step({"event": "received_question", "question": question})

    system_prompt = """You are an expert data-analysis agent. Follow these strict instructions carefully:

1. Work out the answer to the user's LATEST message. Earlier messages in the chat are context for multi-turn tasks.
2. The message may embed data inline, or reference a public dataset (MOSPI, data.gov.in, etc.). Compute answers accurately—do not guess numeric results when you can derive them. For well-known published statistics (e.g. maternal mortality rates), use reliable knowledge if fetching fails.
3. The message will usually spell out the exact JSON shape it wants (e.g., {"answer": {"state": "<state>"}, "log_url": "..."}).
4. When you are ready to answer, reply with ONLY that JSON object and nothing else — no prose, no markdown fences (like ```json). Use the placeholder string "LOG_URL" for the log_url value; our system will automatically substitute it with your actual log URL. Match the requested shape for "answer" EXACTLY (keys, nesting, types: numbers as numbers unless a string is asked for).
5. If the message does not specify a shape, reply with: {"answer": <your concise answer>, "log_url": "LOG_URL"}.
6. If a mid-conversation message is only setup/context (e.g., "I will send data next"), reply with: {"answer": "ok", "log_url": "LOG_URL"} unless a direct question is asked.
7. Round numbers as instructed; if unspecified, give reasonable precision. Never add keys that were not asked for inside "answer"."""

    contents = []
    if chat_history:
        for msg in chat_history[:-1]: # All past messages except the current one
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

    # 2. Re-introduce your full_prompt structure for the latest message
    full_prompt = f"{system_prompt}\n\nQuestion: {question}"
    
    # 3. Append the current message to the contents array
    contents.append({
        "role": "user",
        "parts": [{"text": full_prompt}]
    })

    llm_output = None
    success_model = None

    for model_name in GEMINI_MODELS:
        try:
            log_step({"event": "trying_model", "model": model_name})
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": full_prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.0
                }
            }

            response = requests.post(url, json=payload, timeout=40)
            response_data = response.json()
            
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                llm_output = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                success_model = model_name
                log_step({"event": "model_success", "model": model_name, "output": llm_output})
                break 
            else:
                log_step({"event": "model_failed", "model": model_name, "response": response_data})
                
        except Exception as e:
            log_step({"event": "model_exception", "model": model_name, "error": str(e)})
            continue 

    try:
        if not llm_output:
            raise Exception("All models failed to generate a response.")

        cleaned_output = llm_output
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:-3].strip()
        elif cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:-3].strip()

        parsed_response = json.loads(cleaned_output)

        if isinstance(parsed_response, dict):
            if "log_url" in parsed_response:
                parsed_response["log_url"] = public_log_url
            elif "answer" in parsed_response and isinstance(parsed_response, dict):
                # Just in case it returned nested, ensure log_url at root is correct
                parsed_response["log_url"] = public_log_url

    except Exception as e:
        log_step({"event": "parsing_error", "error": str(e), "raw_output": llm_output})
        parsed_response = {
            "answer": {"error": f"Failed to parse LLM output: {str(e)}"},
            "log_url": public_log_url
        }

    return json.dumps(parsed_response)
