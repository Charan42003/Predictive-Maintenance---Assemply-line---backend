# backend/llm_logic.py
import os
import json
from google import genai
from google.genai.errors import APIError
from PIL import Image
from io import BytesIO

# --- GEMINI CLIENT INITIALIZATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found. Using Mock LLM Responses.")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
MODEL_FLASH = "gemini-2.5-flash"
MODEL_VISION = "gemini-2.5-flash"

# --- MOCKING CLASS FOR LOCAL TESTING WITHOUT KEY ---
class MockLLMService:
    @staticmethod
    def predictive_analysis(logs: list, current_level: str):
        # Analyze the trend to provide context-aware responses
        if not logs:
            return {"root_cause": "Nominal", "fix_plan": ["Monitor system"], "downtime_prediction": "No predicted failure."}
            
        latest_log = logs[-1] if logs else {}
        
        # Dynamic response based on severity level and parameter trends
        if current_level == "EARLY_WARNING":
            # Check which parameter is causing early warning
            if latest_log.get('T', 0) > 202.00:
                return {
                    "root_cause": "Early Thermal Rise Detected",
                    "fix_plan": ["Monitor temperature trend.", "Check cooling system efficiency.", "Schedule preventive maintenance."],
                    "downtime_prediction": "Potential thermal overload in 48-72 hours if trend continues."
                }
            elif latest_log.get('V', 24) < 23.00:
                return {
                    "root_cause": "Voltage Slight Drop",
                    "fix_plan": ["Check power supply stability.", "Inspect cable connections.", "Monitor voltage fluctuations."],
                    "downtime_prediction": "Possible wire feed issues in 24-48 hours."
                }
            elif latest_log.get('Vib', 0) > 1.55:
                return {
                    "root_cause": "Increased Vibration Levels",
                    "fix_plan": ["Check mounting bolts.", "Inspect motor bearings.", "Monitor vibration pattern."],
                    "downtime_prediction": "Bearing failure possible in 1-2 weeks."
                }
            else:
                return {
                    "root_cause": "Early Anomaly Detection",
                    "fix_plan": ["Continue monitoring.", "Review parameter trends.", "Schedule inspection."],
                    "downtime_prediction": "Maintenance required within 1 week."
                }
                
        elif current_level == "WARNING":
            if latest_log.get('T', 0) > 205.00:
                return {
                    "root_cause": "Confirmed Thermal Overload",
                    "fix_plan": ["Reduce operating load.", "Clean cooling system.", "Check thermal sensors."],
                    "downtime_prediction": "Thermal shutdown likely in 2-4 hours."
                }
            elif latest_log.get('V', 24) < 21.50:
                return {
                    "root_cause": "Voltage Instability",
                    "fix_plan": ["IMMEDIATE: Check main power supply.", "Inspect transformer.", "Verify ground connections."],
                    "downtime_prediction": "Wire feed motor failure in 4-6 hours."
                }
            elif latest_log.get('Vib', 0) > 1.70:
                return {
                    "root_cause": "High Vibration - Mechanical Stress",
                    "fix_plan": ["Stop machine for inspection.", "Check alignment.", "Inspect drive components."],
                    "downtime_prediction": "Mechanical failure in 6-8 hours."
                }
            else:
                return {
                    "root_cause": "Multiple Parameter Warning",
                    "fix_plan": ["Stop and inspect machine.", "Review all sensor readings.", "Contact maintenance team."],
                    "downtime_prediction": "System failure in 4-8 hours."
                }
                
        elif current_level == "CRITICAL":
            if latest_log.get('T', 0) > 215.00:
                return {
                    "root_cause": "CRITICAL: Thermal Runaway",
                    "fix_plan": ["EMERGENCY SHUTDOWN.", "Activate fire suppression.", "Call technician immediately."],
                    "downtime_prediction": "IMMEDIATE FAILURE - Downtime 24+ hours."
                }
            elif latest_log.get('V', 24) < 19.00:
                return {
                    "root_cause": "CRITICAL: Power Failure",
                    "fix_plan": ["EMERGENCY STOP.", "Check circuit breakers.", "Verify emergency power."],
                    "downtime_prediction": "IMMEDIATE DOWNTIME - Electrical system damage."
                }
            elif latest_log.get('Vib', 0) > 2.20:
                return {
                    "root_cause": "CRITICAL: Mechanical Failure",
                    "fix_plan": ["EMERGENCY SHUTDOWN.", "Secure area.", "Call mechanical engineer."],
                    "downtime_prediction": "Catastrophic failure imminent - 12+ hour downtime."
                }
            else:
                return {
                    "root_cause": "CRITICAL: System Failure",
                    "fix_plan": ["EMERGENCY PROCEDURE ACTIVATED.", "Isolate machine.", "Call maintenance team (P0)."],
                    "downtime_prediction": "IMMEDIATE FAILURE - Downtime 12+ hours."
                }
                
        return {"root_cause": "Nominal", "fix_plan": ["Monitor"], "downtime_prediction": "Nominal: No predicted failure."}

    @staticmethod
    def check_similarity_and_report():
        return "Historical similarity check requires LLM integration (currently N/A in mock)."
    
    @staticmethod
    def vision_ai_report():
         return {
            "llm_report": {
                "Grade": "C (Needs Attention)", 
                "Description": "Weld bead shows signs of porosity and excessive spatter due to low Gas Flow.",
                "Temperature Range to maintain": "190°C - 200°C",
                "Voltage range": "21.00V - 23.00V",
                "Wire Feed range": "8.50m/min - 9.50m/min",
                "Gas Flow range": "14.00L/min - 16.00L/min"
            },
            "annotated_url": "/mock/annotated_image.png"
        }

# --- 1. Predictive Analysis & Downtime Prediction ---
async def predictive_analysis(logs: list, current_level: str):
    if not client:
        return MockLLMService.predictive_analysis(logs, current_level)

    log_text = "\n".join([f"Time:{l.get('timestamp')}, V:{l.get('V')}, T:{l.get('T')}, G:{l.get('G')}, WQ:{l.get('WQ')}, WF:{l.get('WF')}, Vib:{l.get('Vib')}, Status:{l.get('V_status')}" for l in logs])

    prompt = f"""
    You are an expert Predictive Maintenance AI for a Robotic Welding Machine. The current severity level is: {current_level}.
    Analyze the recent log trend below to:
    1. Determine the Root Cause based on parameter deviations
    2. Provide a 3-step Fix Plan appropriate for {current_level} level
    3. Predict the estimated time until failure with realistic timelines

    For EARLY_WARNING: Provide preventative actions with 24-72 hour predictions
    For WARNING: Provide immediate actions with 2-8 hour predictions  
    For CRITICAL: Provide emergency actions with imminent failure predictions

    Log Data:\n---\n{log_text}\n---\n
    
    RESPOND IN STRICT JSON FORMAT ONLY: 
    {{"root_cause": "...", "fix_plan": ["step 1", "step 2", "step 3"], "downtime_prediction": "Realistic timeline based on severity"}}
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_FLASH,
            contents=[prompt],
            config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except APIError as e:
        print(f"Gemini API Error in Prediction: {e}")
        return MockLLMService.predictive_analysis(logs, current_level)


# --- 2. Similarity Check ---
async def check_similarity_and_report(new_issue_description: str, historical_tickets: list):
    if not client:
        return MockLLMService.check_similarity_and_report()
        
    # Full Gemini logic would be here...
    return MockLLMService.check_similarity_and_report()

# --- 3. Vision AI Analysis (Weld Quality) ---
async def vision_ai_report(image_bytes: bytes):
    if not client:
        return MockLLMService.vision_ai_report()

    # 1. Convert bytes to PIL Image for Gemini
    weld_image = Image.open(BytesIO(image_bytes))

    prompt = """
    Analyze this robotic welding image of a gearbox component. Identify any weld defects (e.g., Porosity, Undercut, Cold Start, Spatter). 
    Provide an analysis and recommend optimal Welding Procedure Specification (WPS) parameters.
    
    RESPOND IN STRICT JSON FORMAT ONLY. The keys MUST match the required format: 
    {{"Grade": "A/B/C/F", "Description": "short analysis of defects (under 20 words)", "Temperature Range to maintain": "min°C - max°C", "Voltage range": "minV - maxV", "Wire Feed range": "min/max m/min", "Gas Flow range": "min/max L/min"}}
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_VISION,
            contents=[weld_image, prompt],
            config={"response_mime_type": "application/json"}
        )
        # Mocking image annotation URL for simplicity
        annotated_image_url = "/mock/annotated_image.png"
        return {
            "llm_report": json.loads(response.text),
            "annotated_url": annotated_image_url
        }
    except APIError as e:
        print(f"Gemini API Error in Vision: {e}")
        # GUARANTEED DICT RETURN ON FAILURE
        return MockLLMService.vision_ai_report()