import pandas as pd
import joblib
import numpy as np
import os

BASE_PATH = r"C:\Users\ramak\OneDrive\Documents\projects\Recommendation System"
TEST_DATA_PATH = os.path.join(BASE_PATH, "data", "processed", "test.csv")
MODEL_PATH = os.path.join(BASE_PATH, "models", "recommender_v1.joblib")

def run_evaluation():
    print("🧪 Loading model and test set...")
    model_data = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    user_emb = model_data["user_embeddings"]
    item_emb = model_data["item_embeddings"]
    
    # We'll check the top 100 users in the test set to get a quick 'Hit Rate'
    sample_users = test_df['user_idx'].unique()[:100]
    hits = 0
    
    print(f"Evaluating {len(sample_users)} users...")
    for user in sample_users:
        # What they actually interacted with in the future (test set)
        actual_items = set(test_df[test_df['user_idx'] == user]['item_idx'])
        
        # What our model predicts
        user_vec = user_emb[user].reshape(1, -1)
        scores = np.dot(user_vec, item_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:10] # Top 10 recommendations
        
        # If any recommended item is in the actual items list, it's a 'Hit'
        if any(rec in actual_items for rec in top_indices):
            hits += 1
            
    hit_rate = (hits / len(sample_users)) * 100
    print(f"\n🎯 Hit Rate @ 10: {hit_rate:.2f}%")
    print("Interpretation: In 10 recommendations, how often did we pick at least one item the user actually bought later.")

if __name__ == "__main__":
    run_evaluation()