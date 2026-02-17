import joblib
import numpy as np
import pandas as pd
import gzip
import json

METADATA_PATH = "data/raw data/meta_Electronics.json.gz" # Update to your metadata filename

def get_metadata_titles(asin_list):
    """Searches the metadata file for specific ASINs."""
    titles = {}
    with gzip.open(METADATA_PATH, 'rb') as f:
        for line in f:
            data = json.loads(line)
            if data['asin'] in asin_list:
                titles[data['asin']] = data.get('title', 'Unknown Title')
            if len(titles) == len(asin_list): # Optimization: stop early if found all
                break
    return titles

def predict_for_user(user_index, k=5):
    # 1. Load Artifacts
    artifacts = joblib.load(r"C:\Users\ramak\OneDrive\Documents\projects\Recommendation System\models\recommender_v1.pkl")
    item_map = artifacts["item_map"] # {asin: index}
    item_embeddings = artifacts["item_embeddings"]
    user_embeddings = artifacts["user_embeddings"]

    # 2. Invert Item Map to get {index: asin}
    index_to_asin = {v: k for k, v in item_map.items()}

    # 3. Calculate Scores
    user_vec = user_embeddings[user_index].reshape(1, -1)
    scores = np.dot(user_vec, item_embeddings.T).flatten()
    
    # 4. Get Top K indices
    top_indices = np.argsort(scores)[::-1][:k]
    
    # 5. Convert Indices to ASINs
    top_asins = [index_to_asin[idx] for idx in top_indices]
    
    # 6. Lookup Titles
    titles_map = get_metadata_titles(top_asins)
    
    print(f"\n🚀 Top {k} Recommendations for User {user_index}:")
    for i, asin in enumerate(top_asins):
        title = titles_map.get(asin, "Title not found in metadata")
        print(f"{i+1}. {title} (ID: {asin})")

if __name__ == "__main__":
    predict_for_user(user_index=0, k=5)