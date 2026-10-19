import os
from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from app.config import settings



@lru_cache()
def clients():
    if not settings.openrouter_api_key or not settings.pinecone_api_key:
        raise ValueError("OpenAI and Pinecone API keys must be set in the environment variables.")

    ai = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key 
    )

    pc = Pinecone(api_key=settings.pinecone_api_key)
    return ai, pc


def ensure_index():
    _, pc = clients()
    if not pc.has_index(settings.pinecone_index):
        pc.create_index(name=settings.pinecone_index, dimension=2048, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    index = pc.Index(settings.pinecone_index)

    description = pc.describe_index(settings.pinecone_index)

    if description.dimension != 2048:
        raise ValueError(
            f"Pinecone index '{settings.pinecone_index}' has "
            f"dimension {description.dimension}, but the embedding "
            "model returns 2048 dimensions. Create a new index "
            "with dimension=2048 and re-ingest your documents."
        )
    return index


def embed(texts):
    ai, _ = clients()

    response = ai.embeddings.create(
        model=settings.openrouter_embed_model,
        input=texts
    )
    return [item.embedding for item in response.data]



def search_kb(query, top_k=5):
    index = ensure_index()
    vector = embed([query])[0]
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [{'page': m.metadata.get('page'), 'text': m.metadata.get('text',''), 'score': round(m.score,3), 'source': 'Gale Encyclopedia (2002)'} for m in result.matches if m.score >= 0.30]