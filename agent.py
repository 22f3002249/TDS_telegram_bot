import os
import json
import requests

GEMINI_MODELS = [
    os.environ.get("PRIMARY_MODEL", "gemini-3.6-flash"),       # Primary Model
    os.environ.get("FALLBACK_MODEL_1", "gemini-3.5-flash-lite"),       # Fallback 1
    os.environ.get("FALLBACK_MODEL_2", "gemini-3.1-flash-lite"),     # Fallback 2
]

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 

LOG_FILE = "run.jsonl"

def log_step(step_data: dict):
    """Append a step to the local JSONL log."""
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(step_data) + "\n")

def run_agent(question: str, public_log_url: str) -> str:
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    log_step({"event": "received_question", "question": question})

    system_prompt = (
        "You are an expert data analyst agent. You will receive a data analysis question. "
        "Analyze the requirements, fetch any necessary public datasets if pointed to, "
        "and return ONLY a valid JSON string matching the requested output format. "
        "Do not include markdown code block ticks like ```json in your response, just the raw JSON object."
    )

    full_prompt = f"{system_prompt}\n\nQuestion: {question}"

    llm_output = None
    success_model = None

    # Try primary model and then fallbacks sequentially
    for model_name in GEMINI_MODELS:
        try:
            log_step({"event": "trying_model", "model": model_name})
            
            # Google AI Studio REST API endpoint
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": full_prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.0
                }
            }

            response = requests.post(url, json=payload, timeout=45)
            response_data = response.json()
            
            # Extract response from Gemini structure
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                llm_output = response_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                success_model = model_name
                log_step({"event": "model_success", "model": model_name, "output": llm_output})
                break # Exit loop if successful
            else:
                log_step({"event": "model_failed", "model": model_name, "response": response_data})
                
        except Exception as e:
            log_step({"event": "model_exception", "model": model_name, "error": str(e)})
            continue # Try next fallback

    # Process the output
    try:
        if not llm_output:
            raise Exception("All Gemini models failed to generate a response.")

        # Clean up potential markdown formatting from LLM response
        cleaned_output = llm_output
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:-3].strip()
        elif cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:-3].strip()

        parsed_answer = json.loads(cleaned_output)
    except Exception as e:
        log_step({"event": "parsing_error", "error": str(e), "raw_output": llm_output})
        parsed_answer = {"error": f"Failed to parse LLM output: {str(e)}"}

    # Construct the final required response object
    final_response = {
        "answer": parsed_answer,
        "log_url": public_log_url
    }

    log_step({"event": "final_response", "model_used": success_model, "response": final_response})
    return json.dumps(final_response)