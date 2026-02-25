import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import os

# Update these to match your actual local paths
BASE_PATH = r"C:\Users\ramak\OneDrive\Documents\projects\Recommendation System"
TRAIN_DATA_PATH = os.path.join(BASE_PATH, "data", "processed", "train.csv")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "models", "recommender_v1.joblib")

def train_model():
    # 1. Load Processed Data
    print(f"Reading training data from {TRAIN_DATA_PATH}...")
    if not os.path.exists(TRAIN_DATA_PATH):
        print("❌ Error: train.csv not found! Run preprocess.py first.")
        return

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # 2. Create Sparse Matrix
    # We explicitly define the shape to avoid index mismatch errors
    n_users = train_df['user_idx'].max() + 1
    n_items = train_df['item_idx'].max() + 1
    
    print(f"Creating {n_users}x{n_items} sparse matrix...")
    user_item_matrix = csr_matrix(
        (train_df['overall'], (train_df['user_idx'], train_df['item_idx'])),
        shape=(n_users, n_items)
    )
    
    # 3. Train SVD
    print("Training SVD model (Matrix Factorization)...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_embeddings = svd.fit_transform(user_item_matrix)
    item_embeddings = svd.components_.T

    # Create the item map
    unique_asins = train_df['asin'].unique()
    item_map = {asin: i for i, asin in enumerate(unique_asins)}

    # Create the user map
    unique_users = train_df['user_idx'].unique()
    user_map = {user: i for i, user in enumerate(unique_users)}

    # 4. Save Artifacts using Joblib (better for large matrices than pickle)
    print("Saving model artifacts...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    artifacts = {
        "user_embeddings": user_embeddings,
        "item_embeddings": item_embeddings,
        "svd_model": svd,
        "user_map": user_map,   # Dictionary: {original_id: index}
        "item_map": item_map    # Dictionary: {asin: index}
    }
    
    joblib.dump(artifacts, MODEL_SAVE_PATH)
    print(f"✅ Done! Model saved to: {MODEL_SAVE_PATH}")

def get_popular_items():
    # Load the training data to calculate popularity
    train_df = pd.read_csv("data/processed/train.csv")

    # Count how many times each item appears and get the top 20
    popular_items = train_df['item_idx'].value_counts().head(20).index.tolist()

    # Save this list
    joblib.dump(popular_items, "models/popular_items.joblib")
    print(f"✅ Saved {len(popular_items)} popular items as fallback.")

if __name__ == "__main__":
    train_model()
    get_popular_items()