from fastapi import FastAPI, Query

from src.llm.llm_provider import get_llm
from src.service.graph.router import start_graph, start_graph_v1, start_graph_v2
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from src.usecase import report_uc as report_use_case

app = FastAPI(title="MACONDO-BE")

class ChatRequest(BaseModel):
  message: str
  session_id: str = "default"  # optional default

@app.get("/")
async def root():
    return "Macondo-be is working"

@app.post("/chat/")
async def chat_endpoint(req: ChatRequest):
  result = await start_graph_v2(req.message)

  return {"reply": f"{result}"}

# ===== REPORT ENDPOINTS =====
@app.post("/report/")
async def save_report(file: UploadFile = File(...),
    ticker: str = Query(...),
    date: str = Query(...),
):
  content_type = file.content_type
  file = await file.read()

  print(f"ENDPOINT: /report/ save report {ticker}")

  metadata = {"ticker": ticker, "date": date}
  report_use_case.save_report(file, metadata, content_type)
  return {"status": "File uploaded successfully"}

@app.get("/report/")
async def get_all_reports():
  print("GET /report/")
  reports = report_use_case.get_report_list()
  return {"reports": reports}

@app.delete("/report/")
async def delete_report(ticker: str):
  print("DELETE /report/")
  report_use_case.delete_report(ticker)
  return {}


# Define request body model
class Ask(BaseModel):
  ask: str

# endpoint to check llm connection
@app.post("/ask/")
async def ask(ask: Ask):
  print("POST /ask/")
  llm = get_llm()
  return llm.invoke(ask.ask).content
