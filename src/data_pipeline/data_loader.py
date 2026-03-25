import pandas as pd
import os

class DataLoader:
    def __init__(self, movies_path: str, credits_path: str):
        """Initializes the DataLoader with paths to the raw datasets."""
        self.movies_path = movies_path
        self.credits_path = credits_path

    def load_and_merge(self) -> pd.DataFrame:
        """Loads the raw CSV files, merges them, and returns a unified DataFrame."""
        print("[INFO] Loading raw datasets...")
        try:
            movies = pd.read_csv(self.movies_path)
            credits = pd.read_csv(self.credits_path)
        except FileNotFoundError as e:
            print(f"[ERROR] Could not find datasets. Check your data/raw/ folder!\n{e}")
            return None

        print("[INFO] Merging datasets on 'title'...")
        # Inner join: Only keep rows where the title exists in both datasets
        movies = movies.merge(credits, on='title')
        return movies

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes noise and keeps only the features necessary for ML."""
        print("[INFO] Cleaning data and dropping noisy columns...")
        
        # We only keep features that define the 'soul' of the movie
        columns_to_keep = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']
        df = df[columns_to_keep]

        # Drop any movies that don't have a plot summary (overview)
        df.dropna(inplace=True)

        return df

    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Saves the clean DataFrame to the processed folder."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Clean data saved to {output_path}")

# --- Quick Test Block ---
if __name__ == "__main__":
    # Define absolute or relative paths based on where the script is run
    # Using relative paths from the root directory 'cinema-sense'
    MOVIES_FILE = "data/raw/tmdb_5000_movies.csv"
    CREDITS_FILE = "data/raw/tmdb_5000_credits.csv"
    OUTPUT_FILE = "data/processed/clean_movies.csv"

    loader = DataLoader(MOVIES_FILE, CREDITS_FILE)
    
    # 1. Load and Merge
    raw_df = loader.load_and_merge()
    
    if raw_df is not None:
        # 2. Clean the Data
        clean_df = loader.clean_data(raw_df)
        print(f"[INFO] Final dataset shape: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns")
        
        # 3. Save it for the next phase
        loader.save_processed_data(clean_df, OUTPUT_FILE)