import numpy as np
import json
import os

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def register_embedding(embedding: np.array, labels: str):
    if not os.path.exists('database.json'):
        database = {}
    else:
        with open('database.json', 'r') as f:
            database = json.load(f)
    
    database[labels] = embedding.tolist()

    with open('database.json', 'w') as f:
        json.dump(database, f)

def recognize_embedding(embedding: np.array, threshold: float = 0.5):
    if not os.path.exists('database.json'):
        return None
    with open('database.json', 'r') as f:
        database = json.load(f)
    
    pred = None
    max_similarity = 0
    for label, register_embedding in database.items():
        similarity = cosine_similarity(embedding, np.array(register_embedding))
        if max_similarity < similarity and similarity > threshold:
            max_similarity = similarity
            pred = label
    
    return pred