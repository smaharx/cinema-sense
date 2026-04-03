import os
# 1. Use the backup mirror server
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 2. Force Python to wait up to 5 minutes before timing out
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300" 

from sentence_transformers import SentenceTransformer

# 3. Load the pre-trained Brain
print("Loading the AI Model... (Extended timeout activated. Let it run!)")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. The text we want the AI to understand
text = "A guy gets stuck in space and grows potatoes"

# 5. Tell the AI to convert the text into math (a vector)
vector = model.encode(text)

# 6. Show the results!
print(f"\nSuccess! The AI turned our sentence into a vector with {len(vector)} dimensions.")
print(f"Here is a peek at the first 5 numbers: {vector[:5]}")