import sys
import os

# Resolve project root (parent of app/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
from schemas import DiabetesInput

# ============================================
# Load Model & Scaler
# ============================================
with open(os.path.join(BASE_DIR, "model", "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# ============================================
# Create FastAPI App
# ============================================
app = FastAPI(title="🩺 Diabetes Prediction API")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ============================================
# Home Page
# ============================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ============================================
# API Endpoint (JSON)
# ============================================
@app.post("/predict")
async def predict(data: DiabetesInput):
    input_data = np.array([[
        data.Pregnancies, data.Glucose, data.BloodPressure,
        data.SkinThickness, data.Insulin, data.BMI,
        data.DiabetesPedigreeFunction, data.Age
    ]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    return {
        "prediction": int(prediction),
        "result": "⚠️ Diabetic" if prediction == 1 else "✅ Not Diabetic",
        "confidence": round(float(max(probability)) * 100, 2),
        "probabilities": {
            "not_diabetic": round(float(probability[0]) * 100, 2),
            "diabetic": round(float(probability[1]) * 100, 2)
        }
    }

# ============================================
# Form Submission (from HTML page)
# ============================================
@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(request: Request):
    form = await request.form()

    input_data = np.array([[
        int(form["Pregnancies"]), float(form["Glucose"]),
        float(form["BloodPressure"]), float(form["SkinThickness"]),
        float(form["Insulin"]), float(form["BMI"]),
        float(form["DiabetesPedigreeFunction"]), int(form["Age"])
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    result = {
        "prediction": "⚠️ Diabetic" if prediction == 1 else "✅ Not Diabetic",
        "confidence": round(float(max(probability)) * 100, 2),
        "prob_no": round(float(probability[0]) * 100, 2),
        "prob_yes": round(float(probability[1]) * 100, 2),
    }

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result,
        "form_data": dict(form)
    })

# ============================================
# ⬇️ THIS IS NEW — Add at the VERY END
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)