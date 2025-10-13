import time
import asyncio
from random import uniform
import json
import os 
from bson.objectid import ObjectId

from db import logs_collection, tickets_collection, get_all_tickets, get_latest_logs 
from llm_logic import predictive_analysis, check_similarity_and_report

# --- RANGES and global state ---
RANGES = {
    "V": {"unit": "V", "safe": (20.00, 24.00), "warning": (18.00, 19.99), "critical": (0.00, 18.00)},
    "T": {"unit": "°C", "safe": (180.00, 200.00), "warning": (200.01, 210.00), "critical": (210.01, 500.00)},
    "G": {"unit": "L/min", "safe": (12.00, 16.00), "warning": (10.00, 11.99), "critical": (0.00, 10.00)},
    "WQ": {"unit": "%", "safe": (95.00, 100.00), "warning": (90.00, 94.99), "critical": (0.00, 90.00)},
    "WF": {"unit": "m/min", "safe": (8.00, 10.00), "warning": (7.00, 7.99), "critical": (0.00, 7.00)},
    "Vib": {"unit": "g", "safe": (0.50, 1.50), "warning": (1.51, 2.00), "critical": (2.01, 5.00)},
}

CURRENT_FAULT = None
SIMULATION_ACTIVE = False
PREDICTIVE_ACTION_STATE = {"active": False, "level": "SAFE", "message": "System nominal.", "ticket_id": None, "prediction": None, "issue_type": None, "last_trigger_time": 0}

# --- UTILITY FUNCTIONS ---

def get_color_status(key, value):
    r = RANGES[key]
    
    # For parameters where HIGH values are bad (Temp, Vibration)
    if key in ["T", "Vib"]:
        if value >= r["critical"][0]:
            return "RED"
        elif r["warning"][0] <= value < r["critical"][0]:
            return "YELLOW" 
        elif r["safe"][1] < value < r["warning"][0]:
            return "ORANGE"
        else:
            return "GREEN"
    
    # For parameters where LOW values are bad (Voltage, Wire Feed, Gas Flow, Weld Quality)
    else:
        if value <= r["critical"][1]:
            return "RED"
        elif r["warning"][1] <= value < r["safe"][0]:
            return "YELLOW"
        elif r["safe"][0] < value <= r["warning"][1]:
            return "ORANGE"
        else:
            return "GREEN"

def get_worst_severity(log):
    """Helper to return the worst severity (RED > YELLOW > ORANGE > GREEN)."""
    worst = "GREEN"
    severity_map = {"RED": 3, "YELLOW": 2, "ORANGE": 1, "GREEN": 0}
    for key in RANGES.keys():
        status = log.get(f'{key}_status', 'GREEN')
        if severity_map[status] > severity_map[worst]:
            worst = status
    return worst

def generate_safe_log():
    v = uniform(*RANGES["V"]["safe"]); t = uniform(*RANGES["T"]["safe"]); g = uniform(*RANGES["G"]["safe"])
    wq = uniform(*RANGES["WQ"]["safe"]); wf = uniform(*RANGES["WF"]["safe"]); vib = uniform(*RANGES["Vib"]["safe"])
    
    return {
        "timestamp": time.time(),
        "V": round(v, 2), "V_status": get_color_status("V", v),
        "T": round(t, 2), "T_status": get_color_status("T", t),
        "G": round(g, 2), "G_status": get_color_status("G", g),
        "WQ": round(wq, 2), "WQ_status": get_color_status("WQ", wq),
        "WF": round(wf, 2), "WF_status": get_color_status("WF", wf),
        "Vib": round(vib, 2), "Vib_status": get_color_status("Vib", vib),
    }

# --- SIMULATION CORE LOGIC ---

async def start_fault_simulation(fault_type: str):
    global CURRENT_FAULT
    CURRENT_FAULT = fault_type
    
    # FAULT TARGETS - CLEAN ISOLATION with proper progression
    fault_targets = {
        "wire_jam": {
            "WF": [7.80, 7.20, 6.50],  # Early → Warning → Critical
        },
        "thermal_spike": {
            "T": [202.00, 205.00, 215.00],  # Early → Warning → Critical
        },
        "voltage_drop": {
            "V": [23.00, 21.00, 18.00],  # Early → Warning → Critical
        },
        "vibration": {
            "Vib": [1.55, 1.70, 2.20],  # Early → Warning → Critical
        },
    }

    if fault_type not in fault_targets:
        CURRENT_FAULT = None
        return

    # Get current state
    latest_log_docs = await get_latest_logs(limit=1)
    if latest_log_docs:
        current_log = latest_log_docs[0].copy()
    else:
        current_log = generate_safe_log()

    TOTAL_STEPS = 90  # Reduced for faster testing
    PHASE_STEPS = 30  # 30 steps per phase
    
    severity_level_map = {"RED": "CRITICAL", "YELLOW": "WARNING", "ORANGE": "EARLY_WARNING", "GREEN": "SAFE"}
    
    # Store initial values for non-affected parameters
    initial_values = {}
    for key in RANGES.keys():
        if key not in fault_targets[fault_type]:
            initial_values[key] = current_log[key]

    for step in range(1, TOTAL_STEPS + 1):
        if CURRENT_FAULT is None:
            print("SIMULATION: Fault resolved manually. Exiting.")
            break
            
        phase = 1 if step <= PHASE_STEPS else (2 if step <= 2 * PHASE_STEPS else 3)
        
        # UPDATE ONLY AFFECTED PARAMETERS
        for key, targets in fault_targets[fault_type].items():
            target_value = targets[phase - 1]
            current_value = current_log[key]
            
            # Calculate step towards target
            remaining_steps = (PHASE_STEPS * phase) - step + 1
            if remaining_steps <= 0:
                remaining_steps = 1
                
            step_val = (target_value - current_value) / remaining_steps
            current_log[key] = round(current_value + step_val, 2)
            current_log[f'{key}_status'] = get_color_status(key, current_log[key])
        
        # KEEP NON-AFFECTED PARAMETERS STABLE
        for key in initial_values:
            # Add very small random variation to make it look natural
            current_log[key] = round(initial_values[key] + uniform(-0.01, 0.01), 2)
            current_log[f'{key}_status'] = get_color_status(key, current_log[key])

        # INSERT SINGLE LOG
        new_log = {
            "timestamp": time.time(),
            **{k: current_log[k] for k in RANGES.keys()},
            **{f'{k}_status': current_log[f'{k}_status'] for k in RANGES.keys()},
        }
        await logs_collection.insert_one(new_log)
        
        current_severity = get_worst_severity(new_log)
        current_level = severity_level_map[current_severity]
        
        # IMMEDIATE TRIGGER - No delays
        should_trigger = (
            current_severity != "GREEN" and 
            current_level != PREDICTIVE_ACTION_STATE["level"]
        )
        
        # Force trigger at phase transitions for critical
        if phase == 3 and step % 5 == 0 and current_severity == "RED":
            should_trigger = True
            
        if should_trigger:
            print(f"IMMEDIATE TRIGGER: {current_level} at step {step}")
            asyncio.create_task(
                trigger_llm_prediction(
                    latest_logs=await get_latest_logs(limit=20), 
                    level=current_level
                )
            )

        await asyncio.sleep(1)

    # Auto-reset after simulation completes
    if CURRENT_FAULT is not None:
        print(f"SIMULATION: {fault_type} completed. Auto-resetting to SAFE.")
        CURRENT_FAULT = None
        # Insert a few safe logs to show recovery
        for _ in range(3):
            safe_log = generate_safe_log()
            await logs_collection.insert_one(safe_log)
            await asyncio.sleep(1)
        
        # Final safe state update
        asyncio.create_task(
            trigger_llm_prediction(
                latest_logs=await get_latest_logs(limit=1), 
                level="SAFE"
            )
        )

# --- MANUAL RESOLUTION ---
async def resolve_current_fault(ticket_id: str):
    global CURRENT_FAULT
    CURRENT_FAULT = None
    
    await tickets_collection.update_one(
        {"_id": ObjectId(ticket_id)},
        {"$set": {"status": "Resolved", "time_resolved": time.time()}}
    )
    print(f"FAULT RESOLVED: Ticket {ticket_id} closed.")

# --- LLM TRIGGER (Optimized for speed) ---
async def trigger_llm_prediction(latest_logs: list, level: str):
    global PREDICTIVE_ACTION_STATE

    # Quick debounce - only 2 seconds for critical
    debounce_time = 2 if level == "CRITICAL" else 5
    if (level != "SAFE" and PREDICTIVE_ACTION_STATE["level"] == level and 
        time.time() - PREDICTIVE_ACTION_STATE["last_trigger_time"] < debounce_time):
        return 

    # 1. Quick LLM Analysis
    analysis = await predictive_analysis(latest_logs, current_level=level)
    issue_type = analysis.get("root_cause", "Nominal" if level == "SAFE" else "Anomaly Detected")
    
    # 2. Auto-Ticket Creation (only for non-SAFE)
    ticket_id = None
    if level != "SAFE":
        new_ticket = {
            "issue_type": issue_type,
            "time_created": time.time(),
            "llm_suggestion": analysis.get('fix_plan', ["Monitor system"]),
            "downtime_prediction": analysis.get('downtime_prediction', "No prediction"),
            "similarity_note": "N/A - Real-time detection",
            "status": "Open",
            "severity": level,
            "associated_logs": [str(log.get('_id', 'no_id')) for log in latest_logs[-5:]],  # Last 5 logs only
        }
        result = await tickets_collection.insert_one(new_ticket)
        ticket_id = str(result.inserted_id)
        print(f"ALERT: {level} Ticket #{ticket_id} created for {issue_type}")

    # 3. Update Global State
    prediction_text = analysis.get('downtime_prediction', "System nominal.")
    
    PREDICTIVE_ACTION_STATE.update({
        "active": level != "SAFE",
        "level": level,
        "message": f"{level}: {issue_type}",
        "ticket_id": ticket_id,
        "prediction": prediction_text,
        "issue_type": issue_type,
        "last_trigger_time": time.time()
    })
    
    if level == "SAFE":
        PREDICTIVE_ACTION_STATE.update({
            'active': False,
            'message': "System nominal.",
            'prediction': "Nominal: No predicted failure.",
            'ticket_id': None,
            'issue_type': "Nominal"
        })
        print("ALERT: System returned to SAFE state.")

# Main simulation loop
async def simulation_loop():
    global SIMULATION_ACTIVE
    SIMULATION_ACTIVE = True
    
    # Initial safe log
    if await logs_collection.count_documents({}) == 0:
        log = generate_safe_log()
        await logs_collection.insert_one(log)
    
    while SIMULATION_ACTIVE:
        if CURRENT_FAULT is None:
            log = generate_safe_log()
            await logs_collection.insert_one(log)
        await asyncio.sleep(2)  # Reduced frequency when in safe mode