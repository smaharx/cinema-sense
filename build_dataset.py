import os
import pandas as pd
from datasets import load_dataset

# Keep our network safeguards on
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

print("Downloading 5,000 real movie plots from Wikipedia...")
# We are pulling the first 5000 rows from a popular Wikipedia movie plot dataset
dataset = load_dataset("vishnupriyavr/wiki-movie-plots-with-summaries", split="train[:5000]")

# Convert the raw data into a Pandas DataFrame (a giant spreadsheet)
df = pd.DataFrame(dataset)

# The dataset has a lot of columns we don't need. Let's filter it down to the core 3.
df = df[['Title', 'Release Year', 'Plot']]

# Rename columns to lowercase to keep our code clean and standardized
df.columns = ['title', 'year', 'plot']

# Create our data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save this cleaned data permanently to a CSV file
df.to_csv("data/movies.csv", index=False)

print(f"\n✅ Success! Saved {len(df)} movies to data/movies.csv")
print("\nHere is a sneak peek at your new massive database:")
print(df.head(3))