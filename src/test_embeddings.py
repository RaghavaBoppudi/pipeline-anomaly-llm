from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI()

# Two sentences that mean the same thing but use different words
sentence_1 = "Pipeline failed due to memory error"
sentence_2 = "DAG crashed because it ran out of RAM"
sentence_3 = "Scheduled report delivered successfully"
sentence_4 = "Memory threshold exceeded in ingestion pipeline"
sentence_5 = "Kafka consumer lag detected on transactions topic"
sentence_6 = "Out of memory exception in data processing job"

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # Cheapest embedding model
        input=text
    )
    return response.data[0].embedding

# Generate embeddings for all three
embedding_1 = get_embedding(sentence_1)
embedding_2 = get_embedding(sentence_2)
embedding_3 = get_embedding(sentence_3)
embedding_4 = get_embedding(sentence_4)
embedding_5 = get_embedding(sentence_5)
embedding_6 = get_embedding(sentence_6)

# Calculate similarity between them
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_1_2 = cosine_similarity(embedding_1, embedding_2)
sim_1_3 = cosine_similarity(embedding_1, embedding_3)

print(f"Sentence 1: {sentence_1}")
print(f"Sentence 2: {sentence_2}")
print(f"Sentence 3: {sentence_3}")
print(f"Sentence 4: {sentence_4}")
print(f"Sentence 5: {sentence_5}")
print(f"Sentence 6: {sentence_6}")
print()
print(f"Similarity 1 vs 4 (same topic, similar words): {cosine_similarity(embedding_1, embedding_4):.4f}")
print(f"Similarity 1 vs 6 (same topic, very similar):  {cosine_similarity(embedding_1, embedding_6):.4f}")
print(f"Similarity 1 vs 5 (different topic):           {cosine_similarity(embedding_1, embedding_5):.4f}")