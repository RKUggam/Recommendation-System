from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os

app = FastAPI()

# 1. Define the path inside the container
MODEL_PATH = "models/recommender_v1.joblib"
REGISTRY_PATH = "models/product_registry.joblib"

# 2. LOAD THE DATA INTO THE 'artifacts' VARIABLE
# This is the line that is likely missing or named incorrectly!
try:
    artifacts = joblib.load(MODEL_PATH)
    registry = joblib.load(REGISTRY_PATH)
    idx_to_asin = {v: k for k, v in artifacts["item_map"].items()}
    print("✅ Model artifacts loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    artifacts = None

@app.get("/recommend/{user_idx}")
def recommend(user_idx: int, k: int = 10):
    # Check if artifacts exists before using
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded on server")
        
    try:
        # The code below uses 'artifacts', so it must match the variable name above
        user_emb = artifacts["user_embeddings"]
        item_emb = artifacts["item_embeddings"]
        
        user_vec = user_emb[user_idx].reshape(1, -1)
        scores = np.dot(user_vec, item_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            asin = idx_to_asin.get(idx, "Unknown")
            results.append({
                "asin": asin, 
                "title": registry.get(asin, "Title Unknown")
            })
        return {"user_idx": user_idx, "recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))