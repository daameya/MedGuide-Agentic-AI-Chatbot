from app.knowledge import embed

vectors = embed([
    "What are the common symptoms of a cold?",
    "How can someone manage a mild cough?"
])

print("Number of vectors:", len(vectors))
print("Vector dimension:", len(vectors[0]))
print("First few values:", vectors[0][:5])