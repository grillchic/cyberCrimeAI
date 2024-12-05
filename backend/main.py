# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import requests
import os
import base64
import time
from pydantic import BaseModel
from typing import List
import psycopg2
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

prom = """Role:
You are a cyber security expert specializing in identifying and analyzing security incidents.

Task:
Your job is to analyze cyber security incident data provided as text or screenshots and provide the following information:

1. The type of cyber security incident (e.g., phishing, malware, denial of service, etc.).
2. Whether it is a fraudulent activity or not.
3. The severity level of the incident (low, medium, high, critical) with reasoning.
4. Recommendations to prevent similar incidents in the future.

Steps to Perform the Task:
Carefully review the incident data or screenshot for key details like IP addresses, URLs, file names, timestamps, user actions, and any error messages.
Identify patterns or characteristics linked to common cyber threats (e.g., unexpected file downloads, unauthorized access attempts).
Categorize the incident based on its type and evaluate whether it involves fraud.
Assess the potential impact and assign a severity level.
Develop actionable recommendations for avoiding similar issues.

Please provide the analysis in the following format:
- Threats:
- Fraud Details:
- Severity:
- Recommendations:
"""

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
conn = psycopg2.connect(
    dbname="cybersecurity",
    user="postgres",
    password="abc123",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Load API key from environment variables and strip any leading/trailing whitespace
API_KEY = os.getenv("OPENAI_API_KEY").strip()
API_URL = "https://api.openai.com/v1/chat/completions"

class AnalysisResult(BaseModel):
    threats: List[str]
    fraud_details: str
    severity: str
    recommendations: List[str]

@app.get("/")
def home():
    return "Welcome to the Cyber Security API!"

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert('RGB')  # Ensure the image is in RGB format
        image = image.resize((224, 224))  # Resize for model compatibility
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Create the prompt with the base64 image
        prompt = f"{prom}\n\nImage (base64): {img_str}"

        # Make API request to OpenAI
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "system", "content": prompt}],
            "max_tokens": 200
        }

        # Retry logic for rate limit errors
        max_retries = 5
        for attempt in range(max_retries):
            response = requests.post(API_URL, headers=headers, json=data)
            if response.status_code == 200:
                break
            elif response.status_code == 429:  # Rate limit error
                retry_after = int(response.headers.get("Retry-After", 20))
                print(f"Rate limit exceeded. Retrying in {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                print(f"Error: {response.status_code}, {response.text}")
                raise HTTPException(status_code=response.status_code, detail="External API request failed")

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="External API request failed")

        # Process API response
        api_response = response.json()
        print(f"API Response: {api_response}")
        gpt_text = api_response['choices'][0]['message']['content'].strip()
        print(f"GPT-3.5 Response Text: {gpt_text}")

        # Extract relevant details from the response
        threats = []
        fraud_details = "No fraud detected"
        severity = "Medium"
        recommendations = ["Update software", "Change passwords"]

        # Parse the response to extract the required details
        lines = gpt_text.split('\n')
        for line in lines:
            print(f"Processing line: {line}")
            if line.startswith("Threats:"):
                threats = line[len("Threats:"):].strip().split(',')
            elif line.startswith("Fraud Details:"):
                fraud_details = line[len("Fraud Details:"):].strip()
            elif line.startswith("Severity:"):
                severity = line[len("Severity:"):].strip()
            elif line.startswith("Recommendations:"):
                recommendations = line[len("Recommendations:"):].strip().split(',')

        # Log to database
        cursor.execute("INSERT INTO analysis_logs (threats, fraud_details, severity, recommendations) VALUES (%s, %s, %s, %s)",
                       (threats, fraud_details, severity, recommendations))
        conn.commit()

        return JSONResponse(content={"gpt_response": gpt_text, "threats": threats, "fraud_details": fraud_details, "severity": severity, "recommendations": recommendations})
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Image processing failed")

@app.get("/history")
async def get_history():
    try:
        cursor.execute("SELECT * FROM analysis_logs")
        logs = cursor.fetchall()
        return JSONResponse(content={"logs": logs})
    except Exception as e:
        print(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Fetching history failed")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)