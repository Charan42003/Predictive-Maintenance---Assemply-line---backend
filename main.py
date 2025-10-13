# backend/main.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time
from bson.objectid import ObjectId 

# Import your modules
from .db import get_latest_logs, seed_initial_tickets, get_all_tickets
from .simulation import simulation_loop, start_fault_simulation, PREDICTIVE_ACTION_STATE, RANGES, resolve_current_fault
from .llm_logic import vision_ai_report

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://polite-river-0f61e460f.2.azurestaticapps.net"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Startup Event: Start the simulation and seed the DB
@app.on_event("startup")
async def startup_event():
    import time 
    await seed_initial_tickets()
    asyncio.create_task(simulation_loop()) 

@app.get("/api/logs/latest")
async def get_logs_data():
    logs = await get_latest_logs(limit=15)
    
    ranges_data = {k: f"{v['unit']} ({v['safe'][0]:.2f}-{v['safe'][1]:.2f})" for k, v in RANGES.items()}
    
    processed_logs = []
    for log in logs:
        display_log = {"timestamp": time.strftime("%H:%M:%S", time.localtime(log["timestamp"]))}
        for key in RANGES.keys():
            display_log[key] = f"{log[key]:.2f}{RANGES[key]['unit']}"
            display_log[f"{key}_status"] = log[f"{key}_status"]
        processed_logs.append(display_log)
    
    return {"logs": processed_logs, "ranges": ranges_data}

@app.post("/api/simulate/{fault_type}")
async def trigger_simulation(fault_type: str):
    # Only allow the 4 enabled buttons
    if fault_type not in ["wire_jam", "thermal_spike", "voltage_drop", "vibration"]:
        raise HTTPException(status_code=400, detail="Invalid fault type. Only wire_jam, thermal_spike, voltage_drop, vibration are supported.")
        
    asyncio.create_task(start_fault_simulation(fault_type))
    return {"message": f"Gradual simulation for {fault_type} STARTED (3 mins). Monitoring engaged."}

@app.get("/api/ticket/history")
async def get_ticket_history():
    tickets = await get_all_tickets()
    
    issue_counts = {}
    for ticket in tickets:
        issue_type = ticket.get("issue_type", "Unknown")
        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

    return {"tickets": tickets, "analytics": {"issue_counts": issue_counts}}

@app.get("/api/alert/latest")
async def get_latest_alert():
    return PREDICTIVE_ACTION_STATE

@app.post("/api/ticket/resolve/{ticket_id}")
async def resolve_ticket(ticket_id: str):
    try:
        if not ObjectId.is_valid(ticket_id):
             raise HTTPException(status_code=400, detail=f"Invalid Ticket ID format: {ticket_id}")
             
        await resolve_current_fault(ticket_id)
        return {"message": f"Ticket {ticket_id} marked as resolved. System returned to SAFE mode."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve ticket: {e}")

@app.post("/api/weld/check-image")
async def process_weld_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG or PNG images are supported.")

    image_bytes = await file.read()
    
    vision_result = await vision_ai_report(image_bytes)
    
    return {
        "status": "success",
        "vision_report": vision_result["llm_report"],
        "annotated_url": vision_result["annotated_url"] 
    }