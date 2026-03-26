import re

class NLPRouter:
    def __init__(self):
        print("[INFO] Initializing Rule-Based NLP Router...")

    def parse_query(self, user_text: str) -> dict:
        """
        Scans a natural language sentence and extracts strict parameters 
        for our Recommender Engine.
        """
        user_text = user_text.lower()
        
        # Default engine parameters
        params = {
            "movie_title": None,
            "max_runtime": 999,  # Default: No limit
            "min_rating": 0.0    # Default: No limit
        }

        # 1. EXTRACT MOVIE TITLE
        # Looks for the phrase "like [Movie]" or "similar to [Movie]"
        title_match = re.search(r'(?:like|similar to) ([a-zA-Z0-9 ]+?)(?: under| with| but| and|$)', user_text)
        if title_match:
            # .title() capitalizes the first letter of each word to match our database (e.g., "the matrix" -> "The Matrix")
            params["movie_title"] = title_match.group(1).strip().title()

        # 2. EXTRACT RUNTIME
        # Looks for phrases like "under 120 mins" or "max 90 minutes"
        runtime_match = re.search(r'(?:under|max|less than) (\d+) (?:min|minutes|mins)', user_text)
        if runtime_match:
            params["max_runtime"] = int(runtime_match.group(1))

        # 3. EXTRACT RATING
        # Qasimio V2 Regex: Looks for "rating" or "rated", ignores whatever messy words 
        # come next (.*?), and grabs the first number it sees (\d+\.?\d*).
        rating_match = re.search(r'(?:rating|rated).*?(\d+\.?\d*)', user_text)
        if rating_match:
            params["min_rating"] = float(rating_match.group(1))

        return params

# --- Quick Test Block ---
if __name__ == "__main__":
    router = NLPRouter()
    
    # Let's test how our system understands human English
    test_queries = [
        "I want a movie like The Matrix",
        "Show me movies similar to Inception but under 120 mins",
        "I need something like Avatar with a rating of at least 7.5 and under 160 minutes"
    ]

    print("\n=== Testing NLP Router ===")
    for query in test_queries:
        print(f"\nUser says: '{query}'")
        extracted_params = router.parse_query(query)
        print(f"Engine translates to: {extracted_params}")