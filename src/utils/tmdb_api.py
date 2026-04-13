import os
import requests
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def fetch_poster(movie_title):
    if not TMDB_API_KEY or "your_key" in TMDB_API_KEY:
        return "https://via.placeholder.com/500x750?text=Missing+API+Key"

    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    
    try:
        response = requests.get(search_url)
        data = response.json()
        
        # DEBUG: Let's see what TMDb is actually saying
        if 'status_message' in data:
            print(f"[TMDb ERROR] {data['status_message']}")
            return "https://via.placeholder.com/500x750?text=Invalid+API+Key"

        if data.get('results') and len(data['results']) > 0:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        
        return "https://via.placeholder.com/500x750?text=No+Poster+Found"
            
    except Exception as e:
        print(f"[SYSTEM ERROR] {e}")
        return "https://via.placeholder.com/500x750?text=System+Error"

if __name__ == "__main__":
    print(f"--- Testing TMDb Connection ---")
    # This will print the first 4 characters of your key to verify it's loading
    print(f"Using API Key starting with: {str(TMDB_API_KEY)[:4]}...") 
    
    poster = fetch_poster("Inception")
    print(f"Resulting URL: {poster}")