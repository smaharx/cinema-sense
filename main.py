import sys
from src.models.hybrid_engine import HybridEngine
import os

def run_cli():
    print("="*60)
    print(" 🎬  WELCOME TO CINEMA-SENSE AI (Terminal Edition)  🎬 ")
    print("="*60)
    print("💡 Pro-tip: Try saying 'I want a movie like Avatar but under 130 mins'")
    print("Type 'exit' or 'quit' to shut down the engine.\n")

    # 1. Boot up the engine
    df_path = "data/processed/movies_with_tags.pkl"
    vectors_path = "data/processed/tfidf_vectors.pkl"
    faiss_path = "data/vector_db/movies.faiss"
    
    try:
        engine = HybridEngine(df_path, vectors_path, faiss_path)
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not load the AI Engine: {e}")
        print("Did you run the data pipeline and preprocessor first?")
        sys.exit(1)

    print("\n[SYSTEM] Engine is online and listening. 🟢")

    # 2. The Interaction Loop
    while True:
        user_input = input("\n🍿 What are you in the mood for?\n> ")
        
        # --- PHASE 1 FIX: Intercept Terminal Commands ---
        if user_input.lower() in ['cls', 'clear']:
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
            
        if user_input.lower() in ['exit', 'quit']:
            print("\nShutting down Cinema-Sense. Catch you next time! ✌️\n")
            break
            
        if not user_input.strip():
            continue
            
        # Get recommendations
        results = engine.get_recommendations(user_input)
        
        # 3. Format the Output beautifully
        print("\n" + "-"*50)
        if 'Error' in results.columns:
            print(f"⚠️  {results['Error'].iloc[0]}")
        elif 'Message' in results.columns:
            print(f"ℹ️  {results['Message'].iloc[0]}")
        else:
            print("✨ TOP RECOMMENDATIONS ✨\n")
            for index, row in results.iterrows():
                # --- PHASE 1 FIX: Human-Readable Match Scores ---
                # Assuming your model returns a distance metric (like Faiss L2 or Cosine distance)
                # where lower is better (e.g., 0.217). We convert it to a percentage (e.g., 78.3%).
                try:
                    raw_score = float(row['similarity_score'])
                    match_pct = max(0, min(100, (1 - raw_score) * 100)) 
                except (ValueError, TypeError):
                    match_pct = 0.0

                print(f"🎥 {row['title']}")
                print(f"   ⭐ Rating: {row['vote_average']}  |  ⏱️ Runtime: {row['runtime']} mins  |  🧬 Match: {match_pct:.1f}%\n")
        print("-"*50)

if __name__ == "__main__":
    run_cli()