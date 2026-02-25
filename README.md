# Amazon Electronics Recommender System 🚀

A high-performance recommendation engine built using **Matrix Factorization (SVD)**. This system predicts user preferences for electronics based on historical interaction data, achieving a **9% Hit Rate @ 10**.

## 📊 Performance
- **Model:** Singular Value Decomposition (SVD)
- **Primary Metric:** 9.00% Hit Rate @ 10
- **Latency:** < 50ms per request (FastAPI + NumPy Dot Product)

## 🛠️ Tech Stack
- **Engine:** Python, Scikit-learn, NumPy, Pandas
- **API:** FastAPI, Uvicorn
- **Deployment:** Docker
- **Serialization:** Joblib

## 🚀 Quick Start (Docker)

Ensure you have Docker Desktop installed and running.

1. **Build the Image:**
   ```bash
   docker build -t recommender-app .

2. **Run the Container:**
    '''bash
    docker run -p 8000:8000 recommender-app

3. **Access the API:**
    Interactive Docs: http://localhost:8000/docs
    Sample Recommendation: http://localhost:8000/recommend/5