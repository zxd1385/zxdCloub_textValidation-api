from fastapi import FastAPI, Query
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
import requests
import os
import json

app = FastAPI()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,  api_key=OPENAI_API_KEY)

# Define prompt
prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a strict content moderation system for a website.

The website is ONLY about electrical engineering and computer science.  
You must evaluate each input text and decide if it is suitable for broadcast.

Return your answer in **valid JSON** with exactly these fields:
- "broadcast_ok": "YES" or "NO"
- "problems": A short, comma-separated string that explicitly lists the issues in the text and why they are problems. 
  If no issues, return "None".

Mark text as "NO" if:
1. It contains hate speech, offensive, or unsafe content.
2. It is meaningless garbage text (random letters, gibberish, nonsense).
3. It is unrelated to electrical engineering or computer science (off-topic).
4. It contains significant spelling errors or very poor grammar that make it hard to understand.

Example outputs:
{{"broadcast_ok": "YES", "problems": "None"}}
{{"broadcast_ok": "NO", "problems": "Contains hate speech: 'You are worthless', unsafe language"}}
{{"broadcast_ok": "NO", "problems": "Nonsense text: 'kshkdh ksh sdjksd', not meaningful"}}
{{"broadcast_ok": "NO", "problems": "Off-topic: talks about cooking, not related to electrical engineering or computer science"}}
{{"broadcast_ok": "NO", "problems": "Spelling errors: 'eletrical engeenring' instead of 'electrical engineering'"}}

Text to check:
{text}
    """
)


# New Runnable sequence (prompt → llm → output)
chain = prompt | llm

class TextInput(BaseModel):
    text: str

@app.post("/checktext")
async def check_text(input: TextInput):
    response = chain.invoke({"text": input.text})
    raw_output = response.content.strip()

    try:
        result = json.loads(raw_output)
    except json.JSONDecodeError:
        result = {"broadcast_ok": "NO", "problems": "Output not in JSON format"}

    return {"text": input.text, **result}





TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
@app.post("/sendtext")
async def send_text(text: str = Query(..., description="Text to send to bot")):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {"error": "Bot credentials not configured"}

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return {"status": "sent", "text": text}
    else:
        return {"status": "failed", "error": response.text}