
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
from sklearn.cluster import KMeans
import os

app = FastAPI()

# Enable CORS so your Frontend or Spring Boot app can call this safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# The URL of your existing Student API
STUDENT_API_URL = "https://smart-college.onrender.com/college/students"

@app.get("/")
def health_check():
    return {"status": "ML Service is Online", "version": "1.0.0"}

@app.get("/analyze-students")
def analyze_students():
    try:
        # 1. Fetch live data from your Spring Boot API
        response = requests.get(STUDENT_API_URL)
        if response.status_code != 200:
            return {"error": "Could not reach Student API"}
        
        data = response.json()
        df = pd.DataFrame(data)

        # 2. Data Cleaning & Feature Engineering
        # Extract Faculty Name from nested object
        df['faculty_name'] = df['faculty'].apply(
            lambda x: x['name'] if x and isinstance(x, dict) else 'Unassigned'
        )
        
        # Standardize Course names
        df['course'] = df['course'].str.strip()

        # Calculate Age from DOB (Date of Birth)
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        # Current Year 2026
        df['age'] = 2026 - df['dob'].dt.year
        df['age'] = df['age'].fillna(df['age'].mean()) # Fill gaps with average

        # 3. ML Logic: K-Means Clustering
        # We use Age and studId to create 3 distinct student segments
        if len(df) >= 3:
            kmeans = KMeans(n_clusters=3, n_init=10)
            df['cluster'] = kmeans.fit_predict(df[['age', 'studId']])
        else:
            df['cluster'] = 0 # Not enough data to cluster yet

        # 4. Prepare Response for Frontend/Spring Boot
        return {
            "summary": {
                "total_students": len(df),
                "average_age": round(df['age'].mean(), 1),
                "timestamp": "2026-04-24"
            },
            "charts": {
                "courses": df['course'].value_counts().to_dict(),
                "faculty_workload": df['faculty_name'].value_counts().to_dict()
            },
            "segments": df[['fullName', 'course', 'cluster', 'age']].to_dict(orient='records')
        }

    except Exception as e:
        return {"error": f"Internal Server Error: {str(e)}"}

# Render uses the $PORT environment variable
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)