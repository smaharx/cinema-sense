import pandas as pd # in this file i am using pandas for merging files, cleaning it, 
                    #removing unwanted movies, specifiying the conditions thourgh columns
import os       #in this files i am using os library to check the file is exist or not
                # if not then create then create new one 

class DataLoader:
    def __init__(self, movies_path: str, credits_path: str):  
        """Using paths to gave to the methods for performing opertaions keeping in
        mind also the reusebality of the files"""
        self.movies_path = movies_path
        self.credits_path = credits_path

    def load_and_merge(self) -> pd.DataFrame:
       """Loading datafiles and merging files based on the title by 
       using inner join """
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
        """Cleaning data by specifying number of columns i want 
        and selecting movies based on summary and time of the each movie given"""
        print("[INFO] Cleaning data and keeping core ML + Metadata columns...")
        
        # ADDED 'vote_average' and 'runtime' for deterministic V2 filtering
        columns_to_keep = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average', 'runtime']
        df = df[columns_to_keep]

        # Drop any movies that don't have a plot summary or missing runtimes
        df.dropna(subset=['overview', 'runtime'], inplace=True)

        return df

    def save_processed_data(self, df: pd.DataFrame, output_path: str):
       """Storing the data into clean file so that we donot need to do all this 
       labour work again and again"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Clean data saved to {output_path}")

# --- Quick Test Block ---
if __name__ == "__main__":
    """assigning paths to the variable so that we can pass them as 
        arguments to the methods"""
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