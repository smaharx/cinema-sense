import pandas as pd #Data handling
import ast # Turning list of dictionaries into workable objects
from sklearn.feature_extraction.text import TfidfVectorizer # Converting text into numbers based on most appeared low priority and less appeared high priority to the words
from nltk.stem.porter import PorterStemmer # 
import nltk
import pickle
import os #Used to confirm files directory like searching, creating 

# Download necessary NLTK data (runs silently if already downloaded)
nltk.download('punkt', quiet=True)

class DataPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        # Max features limits our vector space to the top 5000 most important words
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    def _convert_string_to_list(self, text):
        """Helper to safely evaluate stringified lists from the CSV."""
        try:
            return [i['name'] for i in ast.literal_eval(text)]
        except (ValueError, SyntaxError):
            return []

    def _fetch_director(self, text):
        """Helper to extract only the Director from the crew list."""
        try:
            for i in ast.literal_eval(text):
                if i['job'] == 'Director':
                    return [i['name']]
            return []
        except (ValueError, SyntaxError):
            return []

    def _clean_strings(self, lst):
        """Removes spaces between names (e.g., 'Tom Cruise' -> 'TomCruise') so they are unique tags."""
        return [i.replace(" ", "") for i in lst]

    def _stem_text(self, text):
        """Converts words to their root form (e.g., 'loved', 'loving' -> 'love')."""
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return " ".join(stemmed_words)

    def create_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combines all relevant text features into a single 'tags' column."""
        print("[INFO] Formatting genres, keywords, cast, and crew...")
        
        # Parse stringified lists
        df['genres'] = df['genres'].apply(self._convert_string_to_list)
        df['keywords'] = df['keywords'].apply(self._convert_string_to_list)
        
        # Only take the top 3 cast members
        df['cast'] = df['cast'].apply(self._convert_string_to_list).apply(lambda x: x[:3])
        df['crew'] = df['crew'].apply(self._fetch_director)

        # Remove spaces to create unique entities
        df['genres'] = df['genres'].apply(self._clean_strings)
        df['keywords'] = df['keywords'].apply(self._clean_strings)
        df['cast'] = df['cast'].apply(self._clean_strings)
        df['crew'] = df['crew'].apply(self._clean_strings)

        # Convert overview from string to list
        df['overview'] = df['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

        # Combine everything into a 'tags' column
        print("[INFO] Merging features into 'tags'...")
        df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
        
        # Convert list back to a single string paragraph and make lowercase
        # V2 UPDATE: We must carry vote_average and runtime forward!
        new_df = df[['movie_id', 'title', 'tags', 'vote_average', 'runtime', 'release_date']].copy()
        new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
        
        # Apply stemming
        print("[INFO] Applying stemming to root words...")
        new_df['tags'] = new_df['tags'].apply(self._stem_text)
        
        return new_df

    def vectorize_tags(self, df: pd.DataFrame, output_dir: str):
        """Applies TF-IDF Vectorization and saves the matrix and dataframe."""
        print("[INFO] Vectorizing tags using TF-IDF...")
        
        # This transforms our text into a massive mathematical matrix
        vectors = self.vectorizer.fit_transform(df['tags']).toarray()
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the processed dataframe and the vectors for our ML models
        df_path = os.path.join(output_dir, 'movies_with_tags.pkl')
        vector_path = os.path.join(output_dir, 'tfidf_vectors.pkl')
        
        # --- THE FIX: Add a path and save the Vectorizer itself! ---
        vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
        
        pickle.dump(df, open(df_path, 'wb'))
        pickle.dump(vectors, open(vector_path, 'wb'))
        
        # Saving the "translator dictionary" so main.py can use it later
        pickle.dump(self.vectorizer, open(vectorizer_path, 'wb'))
        
        print(f"[SUCCESS] Vectorization complete. Shape: {vectors.shape}")
        print(f"[SUCCESS] Files saved to {output_dir}")
        print(f"[SUCCESS] Saved Vectorizer model for real-time text translation!")
        return vectors

# --- Quick Test Block ---
if __name__ == "__main__":
    INPUT_FILE = "data/processed/clean_movies.csv"
    OUTPUT_DIR = "data/processed/"

    print("--- Starting Phase 3: NLP Processing ---")
    try:
        clean_df = pd.read_csv(INPUT_FILE)
        
        preprocessor = DataPreprocessor()
        
        # 1. Create the Tags
        tagged_df = preprocessor.create_tags(clean_df)
        
        # 2. Vectorize and Save
        vectors = preprocessor.vectorize_tags(tagged_df, OUTPUT_DIR)
        
    except FileNotFoundError:
        print(f"[ERROR] Could not find {INPUT_FILE}. Did you run data_loader.py first?")