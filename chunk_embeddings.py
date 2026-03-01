"""
Optimized Chunks to Vector DB with Multi-Model Support
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import ollama

# ============================================================================
# EMBEDDING MODEL PROFILES
# ============================================================================

EMBEDDING_MODELS = {
    "fast": {
        "name": "nomic-embed-text",
        "dimension": 768,
        "speed": "⚡ Fast",
        "quality": "⭐⭐⭐",
        "description": "Quick processing, good for most videos",
        "best_for": "General content, quick testing"
    },
    "balanced": {
        "name": "mxbai-embed-large",
        "dimension": 1024,
        "speed": "⚡⚡ Medium",
        "quality": "⭐⭐⭐⭐",
        "description": "Better accuracy, slightly slower",
        "best_for": "Educational content, detailed tutorials"
    },
    "accurate": {
        "name": "bge-large-en-v1.5",
        "dimension": 1024,
        "speed": "⚡⚡⚡ Slow",
        "quality": "⭐⭐⭐⭐⭐",
        "description": "Highest quality, takes longer",
        "best_for": "Technical content, research videos"
    },
    "multilingual": {
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "dimension": 768,
        "speed": "⚡⚡ Medium",
        "quality": "⭐⭐⭐⭐",
        "description": "Supports 50+ languages",
        "best_for": "Non-English or mixed language videos"
    },
    "code": {
        "name": "jina-embeddings-v2-base-code",
        "dimension": 768,
        "speed": "⚡⚡ Medium",
        "quality": "⭐⭐⭐⭐",
        "description": "Optimized for programming tutorials",
        "best_for": "Coding videos, tech talks"
    }
}

# CONFIG
CHUNKS_DIR = Path("chunks")
DB_DIR = Path("vector_db")
COLLECTION_NAME = "video_chunks"
BATCH_SIZE = 50
MAX_WORKERS = 3


# ============================================================================
# MODEL SELECTION
# ============================================================================

def display_embedding_models():
    """Display available embedding models."""
    print("\n" + "="*70)
    print("  EMBEDDING MODEL OPTIONS")
    print("="*70 + "\n")
    
    for i, (key, profile) in enumerate(EMBEDDING_MODELS.items(), 1):
        print(f"{i}. [{key.upper()}] - {profile['name']}")
        print(f"   Speed: {profile['speed']} | Quality: {profile['quality']}")
        print(f"   → {profile['description']}")
        print(f"   💡 Best for: {profile['best_for']}\n")


def select_embedding_model() -> tuple[str, dict]:
    """
    Let user select embedding model.
    Returns (model_name, profile_dict)
    """
    display_embedding_models()
    
    keys = list(EMBEDDING_MODELS.keys())
    default_key = "fast"
    
    print(f"Choose embedding model (1-{len(keys)}) or press Enter for default [{default_key}]: ", end="")
    choice = input().strip()
    
    if not choice:
        selected_key = default_key
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                selected_key = keys[idx]
            else:
                print(f"⚠️  Invalid choice. Using default: {default_key}")
                selected_key = default_key
        except ValueError:
            print(f"⚠️  Invalid input. Using default: {default_key}")
            selected_key = default_key
    
    selected_profile = EMBEDDING_MODELS[selected_key]
    
    print(f"\n✅ Selected: {selected_profile['name']} ({selected_key})")
    print(f"   Dimension: {selected_profile['dimension']} | Speed: {selected_profile['speed']}")
    print("="*70 + "\n")
    
    return selected_profile['name'], selected_profile


# ============================================================================
# CHROMADB INITIALIZATION
# ============================================================================

def init_chromadb(embed_model_name: str, dimension: int):
    """Initialize ChromaDB with selected embedding model."""
    DB_DIR.mkdir(exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Store embedding model info in collection metadata
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": embed_model_name,
            "embedding_dimension": dimension
        },
        embedding_function=None  # Manual embedding with Ollama
    )
    
    return client, collection


# ============================================================================
# EMBEDDING FUNCTIONS
# ============================================================================

def embed_text(text: str, model: str, dimension: int) -> List[float]:
    """Embed single text using Ollama."""
    try:
        response = ollama.embeddings(
            model=model,
            prompt=text
        )
        embedding = response["embedding"]
        
        # Validate dimension
        if len(embedding) != dimension:
            print(f"⚠️  Warning: Expected {dimension}D, got {len(embedding)}D")
        
        return embedding
    except Exception as e:
        print(f"⚠️  Embedding error: {e}")
        return [0.0] * dimension  # Return zero vector on error


def embed_batch(texts: List[str], model: str, dimension: int) -> List[List[float]]:
    """Embed multiple texts."""
    embeddings = []
    for text in texts:
        embeddings.append(embed_text(text, model, dimension))
    return embeddings


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def get_existing_ids(collection) -> set:
    """Get all existing chunk IDs from collection."""
    try:
        result = collection.get()
        return set(result["ids"])
    except Exception:
        return set()


def process_chunks_batch(
    chunks: List[Dict[str, Any]], 
    existing_ids: set,
    collection,
    embed_model: str,
    dimension: int
) -> int:
    """Process chunks in batch for efficiency."""
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    
    if not new_chunks:
        return 0
    
    ids = [c["chunk_id"] for c in new_chunks]
    texts = [c["text"] for c in new_chunks]
    metadatas = [
        {
            "video": c["video"],
            "start": c["start"],
            "end": c["end"],
            "duration": c.get("duration", c["end"] - c["start"]),
            "word_count": c.get("word_count", len(c["text"].split()))
        }
        for c in new_chunks
    ]
    
    # Embed in batch
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = embed_batch(texts, embed_model, dimension)
    
    # Store in ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    return len(new_chunks)


def process_file(
    chunk_file: Path, 
    existing_ids: set,
    collection,
    embed_model: str,
    dimension: int
) -> tuple[str, int]:
    """Process a single chunk file."""
    try:
        with open(chunk_file, encoding="utf-8") as f:
            chunks = json.load(f)
        
        if not chunks:
            return (chunk_file.name, 0)
        
        print(f"🔢 Processing: {chunk_file.name}")
        
        # Process in batches
        total_added = 0
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            added = process_chunks_batch(batch, existing_ids, collection, embed_model, dimension)
            total_added += added
            
            if added > 0:
                existing_ids.update(c["chunk_id"] for c in batch)
        
        status = f"✅ {chunk_file.name}: {total_added} new"
        if total_added < len(chunks):
            status += f" ({len(chunks) - total_added} skipped)"
        
        print(status)
        return (chunk_file.name, total_added)
        
    except Exception as e:
        print(f"❌ Error processing {chunk_file.name}: {e}")
        return (chunk_file.name, 0)


def get_chunk_files() -> List[Path]:
    """Get all chunk files sorted by size."""
    if not CHUNKS_DIR.exists():
        return []
    files = list(CHUNKS_DIR.glob("*_chunks.json"))
    return sorted(files, key=lambda x: x.stat().st_size)


def get_collection_stats(collection, embed_model: str) -> Dict[str, Any]:
    """Get detailed collection statistics."""
    count = collection.count()
    
    if count > 0:
        sample = collection.get(limit=min(100, count))
        videos = set(m["video"] for m in sample["metadatas"])
        
        return {
            "total_chunks": count,
            "unique_videos": len(videos),
            "collection_name": COLLECTION_NAME,
            "embed_model": embed_model
        }
    
    return {
        "total_chunks": 0,
        "unique_videos": 0,
        "collection_name": COLLECTION_NAME,
        "embed_model": embed_model
    }


# ============================================================================
# QUERY FUNCTIONS (for testing)
# ============================================================================

def query_similar(
    query: str, 
    collection,
    embed_model: str,
    dimension: int,
    n_results: int = 5
) -> List[Dict[str, Any]]:
    """Query for similar chunks."""
    query_embedding = embed_text(query, embed_model, dimension)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "video": results["metadatas"][0][i]["video"],
            "timestamp": f"{results['metadatas'][0][i]['start']:.1f}s - {results['metadatas'][0][i]['end']:.1f}s",
            "start": results["metadatas"][0][i]["start"],
            "end": results["metadatas"][0][i]["end"],
            "distance": results["distances"][0][i] if "distances" in results else None
        })
    
    return formatted


def build_llm_context(query_results: List[Dict[str, Any]]) -> str:
    """Build optimized context string for LLM from query results."""
    context_parts = []
    
    for i, result in enumerate(query_results, 1):
        context_parts.append(
            f"[Source {i}] Video: {result['video']} | Time: {result['timestamp']}\n"
            f"{result['text']}\n"
        )
    
    return "\n".join(context_parts)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("  🤖 RAG PIPELINE - EMBEDDING GENERATION")
    print("="*70)
    
    # Select embedding model
    embed_model_name, embed_profile = select_embedding_model()
    dimension = embed_profile["dimension"]
    
    # Initialize database
    client, collection = init_chromadb(embed_model_name, dimension)
    
    # Get chunk files
    chunk_files = get_chunk_files()
    
    if not chunk_files:
        print("❌ No chunk files found")
        return
    
    print(f"📁 Found {len(chunk_files)} file(s)")
    print(f"🔍 Checking existing embeddings...\n")
    
    existing_ids = get_existing_ids(collection)
    print(f"📊 {len(existing_ids)} chunks already in DB\n")
    
    total_added = 0
    
    # Process files sequentially
    for chunk_file in chunk_files:
        _, added = process_file(chunk_file, existing_ids, collection, embed_model_name, dimension)
        total_added += added
    
    stats = get_collection_stats(collection, embed_model_name)
    
    print(f"\n{'='*70}")
    print(f"🎉 Embedding Complete")
    print(f"📊 Total Chunks: {stats['total_chunks']} | Videos: {stats['unique_videos']}")
    print(f"➕ New Chunks Added: {total_added}")
    print(f"🔧 Model Used: {embed_model_name}")
    print(f"{'='*70}")
    
    # Example query demonstration
    if stats['total_chunks'] > 0:
        print("\n" + "="*70)
        print("Example Query")
        print("="*70)
        
        test_query = "What is the main topic?"
        print(f"Query: '{test_query}'")
        print("Embedding query...")
        
        results = query_similar(test_query, collection, embed_model_name, dimension, n_results=3)
        
        print()
        for r in results:
            print(f"[{r['video']}] @ {r['timestamp']}")
            print(f"{r['text'][:150]}...\n")
        
        # Show how to format for LLM
        print("="*70)
        print("LLM Context Format")
        print("="*70)
        llm_context = build_llm_context(results)
        print(llm_context[:500] + "...")


if __name__ == "__main__":
    main()