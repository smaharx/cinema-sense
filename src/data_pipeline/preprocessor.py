"""
Phase 2: Deep Learning NLP Processing
Converts raw movie text into Dense Vectors using HuggingFace BERT.
"""

import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

class Preprocessor:
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir
        
        # --- THE DEEP LEARNING UPGRADE ---
        print("[INFO] Booting HuggingFace SentenceTransformer (all-MiniLM-L6-v2)...")
        # This downloads the pre-trained neural network weights
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[INFO] Formatting metadata into context strings...")
        # Keeping our VIP list, including release_date for our UI filters!
        new_df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average', 'runtime', 'release_date']].copy()
        
        # Combine everything into a single 'tags' column for the AI to read
        # Notice we are keeping the text readable like a sentence, which BERT loves
        new_df['tags'] = (
            new_df['overview'].fillna('') + " " + 
            new_df['genres'].fillna('') + " " + 
            new_df['keywords'].fillna('') + " " + 
            new_df['cast'].fillna('') + " " + 
            new_df['crew'].fillna('')
        )
        
        # Clean up whitespace and lowercase
        new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x.split()).lower())
        return new_df

    def process_and_save(self):
        print(f"[INFO] Loading data from {self.input_path}...")
        try:
            df = pd.read_csv(self.input_path)
        except FileNotFoundError:
            print("[ERROR] Clean CSV not found. Run data_loader.py first.")
            return

        tagged_df = self.create_tags(df)

        print(f"[INFO] Encoding {len(tagged_df)} movies into Dense Vectors...")
        print("[INFO] This will take a moment. Go grab a coffee. ☕")
        
        # 1. ENCODING (The heavy math happens here)
        # We use batch_size=32 to protect your RAM. show_progress_bar gives you a nice visual.
        embeddings = self.model.encode(
            tagged_df['tags'].tolist(), 
            batch_size=32, 
            show_progress_bar=True
        )

        # 2. NORMALIZATION (For Cosine Similarity)
        # Convert to float32 (FAISS requirement) and normalize to length 1
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)

        # 3. FAISS INDEXING
        dimension = embeddings.shape[1] # Should be exactly 384
        print(f"[INFO] Building FAISS Index with {dimension} dimensions...")
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # 4. SAVING
        # Save the DataFrame (for our UI)
        df_path = os.path.join(self.output_dir, "movies_with_tags.pkl")
        tagged_df.to_pickle(df_path)
        
        # Save the FAISS Index (The ultra-fast map)
        faiss_path = os.path.join(self.output_dir, "movies.faiss")
        faiss.write_index(index, faiss_path)
        
        print(f"[SUCCESS] Deep Learning Engine prepared and saved to {self.output_dir}")

# --- Execution Block ---
if __name__ == "__main__":
    INPUT_FILE = "data/processed/clean_movies.csv"
    OUTPUT_DIR = "data/processed"
    
    # We must ensure the FAISS vector db directory exists
    os.makedirs("data/vector_db", exist_ok=True)
    
    processor = Preprocessor(INPUT_FILE, OUTPUT_DIR)
    processor.process_and_save()
    
    # Move the FAISS file to its dedicated folder to keep our architecture clean
    os.replace("data/processed/movies.faiss", "data/vector_db/movies.faiss")