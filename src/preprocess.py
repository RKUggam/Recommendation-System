import pandas as pd

def filter_data(df, min_user_reviews=5, min_item_reviews=5):
    """
    Reduces sparsity by filtering out 'noisy' low-interaction users and items.
    """
    # Filter items
    item_counts = df.groupby('asin').size()
    active_items = item_counts[item_counts >= min_item_reviews].index
    df = df[df['asin'].isin(active_items)]
    
    # Filter users
    user_counts = df.groupby('reviewerID').size()
    active_users = user_counts[user_counts >= min_user_reviews].index
    df = df[df['reviewerID'].isin(active_users)]
    
    return df

def create_mappings(df):
    """
    RecSys models need integer IDs, not strings. 
    This creates a mapping for UserIDs and Asins.
    """
    user_map = {id: i for i, id in enumerate(df['reviewerID'].unique())}
    item_map = {id: i for i, id in enumerate(df['asin'].unique())}
    
    df['user_idx'] = df['reviewerID'].map(user_map)
    df['item_idx'] = df['asin'].map(item_map)
    
    return df, user_map, item_map

def temporal_split(df, test_size=0.2):
    """
    Splits data based on time so we predict the 'future' from the 'past'.
    """
    df = df.sort_values('unixReviewTime')
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    return train_df, test_df