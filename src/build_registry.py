import gzip
import json
import joblib
import os

def build_registry():
    # Update this path to your actual metadata file
    metadata_path = "data/raw data/meta_Electronics.json.gz"
    output_path = "models/product_registry.joblib"
    
    print(f"Reading metadata from {metadata_path}...")
    registry = {}
    
    try:
        with gzip.open(metadata_path, 'rb') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                asin = data.get('asin')
                title = data.get('title', 'Unknown Product')
                
                if asin:
                    registry[asin] = title
                
                if i % 100000 == 0:
                    print(f"Processed {i} items...")
                    
        print(f"Saving registry with {len(registry)} products...")
        joblib.dump(registry, output_path)
        print(f"✅ Registry created successfully at {output_path}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {metadata_path}. Please check the filename.")

if __name__ == "__main__":
    build_registry()