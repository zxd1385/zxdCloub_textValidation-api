from fastapi import FastAPI, Query
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

app = FastAPI()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,  api_key=OPENAI_API_KEY)

# Define prompt
prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a content moderation system for a website.
Decide if the following text is GOOD TO BROADCAST or NOT GOOD TO BROADCAST.
Respond with only one word: "YES" (if good) or "NO" (if not).

Text: {text}
    """
)

# New Runnable sequence (prompt → llm → output)
chain = prompt | llm

@app.get("/checktext")
async def check_text(text: str = Query(..., description="Text to check")):
    response = chain.invoke({"text": text})
    result = response.content.strip()
    return {"text": text, "broadcast_ok": result}
