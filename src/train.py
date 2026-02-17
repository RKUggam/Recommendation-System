import pandas as pd
import joblib # for saving the model
from preprocess import filter_data, create_mappings, temporal_split
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import os

def train_model():
    # 1. Load Data (Using the Electronics data you have)
    print("Loading data...")
    reader = pd.read_json(r"C:\Users\ramak\OneDrive\Documents\projects\Recommendation System\data\raw data\Electronics.json.gz", 
                          lines=True, chunksize=200000)
    raw_df = next(reader)

    # 2. Preprocess
    print("Preprocessing and filtering...")
    df = filter_data(raw_df)
    df, user_map, item_map = create_mappings(df)
    train_df, test_df = temporal_split(df)
    
    # 3. Create Sparse Matrix (Memory efficient way to handle 97% zeros)
    # Rows = Users, Cols = Items, Values = Ratings
    print("Creating sparse matrix...")
    user_item_matrix = csr_matrix(
        (train_df['overall'], (train_df['user_idx'], train_df['item_idx']))
    )
    
    # 4. Train SVD (Matrix Factorization)
    print("Training SVD model...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_embeddings = svd.fit_transform(user_item_matrix)
    item_embeddings = svd.components_.T
    
    # 5. Save Artifacts
    print("Saving model and mappings...")
    os.makedirs(r"C:\Users\ramak\OneDrive\Documents\projects\Recommendation System\models", exist_ok=True)
    # In a real project, we'd save these to be loaded by FastAPI later
    import pickle
    artifacts = {
        "user_embeddings": user_embeddings,
        "item_embeddings": item_embeddings,
        "svd_model": svd,
        "user_map": user_map,
        "item_map": item_map
    }
    with open(r"C:\Users\ramak\OneDrive\Documents\projects\Recommendation System\models\recommender_v1.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    
    print("Done! Model saved to models/recommender_v1.pkl")

if __name__ == "__main__":
    train_model()