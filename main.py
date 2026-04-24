import os
import requests
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
import uvicorn

app = FastAPI()

# Enable CORS so your Spring Boot or React app can call this live
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

EXTERNAL_API = "https://smart-college.onrender.com/college/students"

@app.get("/")
def home():
    return {"message": "Smart College ML Service is Running", "target_api": EXTERNAL_API}

@app.get("/analyze-students")
def analyze():
    try:
        # 1. Fetch Data
        resp = requests.get(EXTERNAL_API).json()
        df = pd.DataFrame(resp)
        
        if df.empty:
            return {"error": "No student data found"}

        # 2. Data Cleaning
        # Safe extraction of faculty name
        df['faculty_name'] = df['faculty'].apply(
            lambda x: x['name'] if x and isinstance(x, dict) else 'Unassigned'
        )
        
        # Clean course names (remove extra spaces/newlines)
        df['course'] = df['course'].astype(str).str.strip()
        
        # Age Calculation (Current Year 2026)
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = 2026 - df['dob'].dt.year
        df['age'] = df['age'].fillna(df['age'].mean())

        # 3. ML Clustering
        # We need at least 3 students to make 3 clusters meaningful
        num_clusters = min(3, len(df))
        if num_clusters > 1:
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
            df['cluster'] = kmeans.fit_predict(df[['age', 'studId']])
        else:
            df['cluster'] = 0

        # 4. Prepare Response for Dashboard
        return {
            "summary": {
                "total": len(df), 
                "avg_age": round(df['age'].mean(), 1)
            },
            "charts": {
                "courses": df['course'].value_counts().to_dict(),
                "faculty": df['faculty_name'].value_counts().to_dict()
            },
            "segments": df[['fullName', 'course', 'cluster', 'age']].to_dict(orient='records')
        }
    except Exception as e:
        return {"error": f"Internal Error: {str(e)}"}

# Render deployment uses the $PORT environment variable
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)