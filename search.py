import os
from sentence_transformers import SentenceTransformer, util

# Keep our network safeguards just in case it checks the cache
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

print("Loading AI Brain... (This will be instant now because it's saved locally!)")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 1. Our Mini Movie Database
movies = [
    {"title": "The Martian", "plot": "An astronaut is stranded on Mars and must use his scientific knowledge to survive and grow food."},
    {"title": "Interstellar", "plot": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."},
    {"title": "The Godfather", "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."},
    {"title": "Jurassic Park", "plot": "Scientists clone dinosaurs for a theme park, but things go wrong and the dinosaurs escape."},
    {"title": "Toy Story", "plot": "A cowboy doll is jealous when a new spaceman action figure becomes the favorite toy in a boy's room."}
]

# Extract just the plots for the AI to read
plots = [movie["plot"] for movie in movies]

print("Vectorizing movie plots... (Building the mathematical database)")
# This turns all 5 plots into 384-dimensional vectors
plot_embeddings = model.encode(plots)

# 2. The User's Search Query (Change this to whatever you want!)
query = "I want to watch something about farming on another planet."
print(f"\nUser Search: '{query}'")

# Turn the user's query into a vector too
query_embedding = model.encode(query)

# 3. The Magic Math (Cosine Similarity)
# This compares the query vector to all the plot vectors and scores them from 0.0 to 1.0
results = util.semantic_search(query_embedding, plot_embeddings)[0]

# 4. Show the Top 3 Results
print("\n🎬 TOP SEARCH RESULTS:")
for i, result in enumerate(results[:3]):
    # result['corpus_id'] tells us which movie in our list it matched with
    movie_index = result['corpus_id']
    score = result['score']
    
    print(f"{i+1}. {movies[movie_index]['title']} (Match Score: {score:.2f})")