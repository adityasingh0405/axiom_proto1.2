"""
unified_query.py - Unified query interface for videos + PDFs
Teach mode only - conversational interface
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings
import ollama
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "vector_db"

# Collections
VIDEO_TEXT_COLLECTION = "video_chunks"
VIDEO_FRAME_COLLECTION = "video_frames"
PDF_TEXT_COLLECTION = "pdf_chunks"
PDF_IMAGE_COLLECTION = "pdf_images"

# Models
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"

# Retrieval settings
TOP_K_VIDEO_TEXT = 5
TOP_K_VIDEO_FRAMES = 2
TOP_K_PDF_TEXT = 5
TOP_K_PDF_IMAGES = 2
MAX_DISTANCE = 260


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_db():
    """Initialize all ChromaDB collections."""
    if not DB_DIR.exists():
        print(f"❌ Database not found at: {DB_DIR}")
        print("💡 Run 'python process_all.py' first")
        sys.exit(1)
    
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Initialize collections
    collections = {}
    counts = {}
    
    # Video text
    try:
        collections['video_text'] = client.get_or_create_collection(
            name=VIDEO_TEXT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )
        counts['video_text'] = collections['video_text'].count()
    except:
        collections['video_text'] = None
        counts['video_text'] = 0
    
    # Video frames
    try:
        collections['video_frames'] = client.get_or_create_collection(
            name=VIDEO_FRAME_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )
        counts['video_frames'] = collections['video_frames'].count()
    except:
        collections['video_frames'] = None
        counts['video_frames'] = 0
    
    # PDF text
    try:
        collections['pdf_text'] = client.get_or_create_collection(
            name=PDF_TEXT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )
        counts['pdf_text'] = collections['pdf_text'].count()
    except:
        collections['pdf_text'] = None
        counts['pdf_text'] = 0
    
    # PDF images
    try:
        collections['pdf_images'] = client.get_or_create_collection(
            name=PDF_IMAGE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )
        counts['pdf_images'] = collections['pdf_images'].count()
    except:
        collections['pdf_images'] = None
        counts['pdf_images'] = 0
    
    # Check if we have any content
    total_content = sum(counts.values())
    if total_content == 0:
        print("❌ No content found in database")
        print("💡 Run 'python process_all.py' first")
        sys.exit(1)
    
    return collections, counts


# ============================================================================
# EMBEDDING & RETRIEVAL
# ============================================================================

def embed_query(query: str) -> List[float]:
    """Embed query using Ollama."""
    try:
        response = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=query
        )
        return response["embedding"]
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None


def search_collection(
    collection,
    query_embedding: List[float],
    top_k: int,
    max_distance: float = MAX_DISTANCE
) -> List[Dict]:
    """Generic search function for any collection."""
    if not collection or not query_embedding:
        return []
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if not results["ids"][0]:
            return []
        
        items = []
        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i] if "distances" in results else 0
            
            if dist > max_distance:
                continue
            
            items.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": dist
            })
        
        return items
        
    except Exception as e:
        print(f"⚠️  Search error: {e}")
        return []


def retrieve_all_content(query: str, collections: Dict) -> Dict[str, List]:
    """Search all collections and return results."""
    query_embedding = embed_query(query)
    
    if not query_embedding:
        return {"video_text": [], "video_frames": [], "pdf_text": [], "pdf_images": []}
    
    results = {
        "video_text": search_collection(
            collections['video_text'],
            query_embedding,
            TOP_K_VIDEO_TEXT
        ),
        "video_frames": search_collection(
            collections['video_frames'],
            query_embedding,
            TOP_K_VIDEO_FRAMES
        ),
        "pdf_text": search_collection(
            collections['pdf_text'],
            query_embedding,
            TOP_K_PDF_TEXT
        ),
        "pdf_images": search_collection(
            collections['pdf_images'],
            query_embedding,
            TOP_K_PDF_IMAGES
        )
    }
    
    return results


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS."""
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


# ============================================================================
# CONTEXT BUILDING
# ============================================================================

def build_context(results: Dict[str, List]) -> str:
    """Build unified context from all sources."""
    context_parts = []
    
    # Video text chunks
    if results["video_text"]:
        context_parts.append("=== VIDEO TRANSCRIPT CONTENT ===\n")
        for item in results["video_text"]:
            meta = item["metadata"]
            ts = format_timestamp(meta.get("start", 0))
            video_name = meta.get("video", "Unknown")
            context_parts.append(
                f"[Video: {video_name} @ {ts}]\n{item['document']}\n"
            )
    
    # Video frames
    if results["video_frames"]:
        context_parts.append("\n=== VIDEO VISUAL CONTENT (Screenshots) ===\n")
        for item in results["video_frames"]:
            meta = item["metadata"]
            ts = format_timestamp(meta.get("timestamp", 0))
            video_name = meta.get("video", "Unknown")
            context_parts.append(
                f"[Video: {video_name} @ {ts} - Visual]\n{item['document']}\n"
            )
    
    # PDF text chunks
    if results["pdf_text"]:
        context_parts.append("\n=== PDF TEXT CONTENT ===\n")
        for item in results["pdf_text"]:
            meta = item["metadata"]
            doc_name = meta.get("document", "Unknown")
            chunk_idx = meta.get("chunk_index", 0)
            context_parts.append(
                f"[PDF: {doc_name} - Chunk {chunk_idx}]\n{item['document']}\n"
            )
    
    # PDF images
    if results["pdf_images"]:
        context_parts.append("\n=== PDF VISUAL CONTENT (Diagrams/Charts) ===\n")
        for item in results["pdf_images"]:
            meta = item["metadata"]
            doc_name = meta.get("document", "Unknown")
            page = meta.get("page", 0)
            context_parts.append(
                f"[PDF: {doc_name} - Page {page} - Visual]\n{item['document']}\n"
            )
    
    return "\n".join(context_parts)


# ============================================================================
# TEACHING MODE
# ============================================================================

def teach(query: str, context: str) -> str:
    """Generate educational explanation using LLM."""
    system_prompt = """You are a precise AI Teaching Assistant with access to educational content from BOTH videos AND PDFs, including visual information.

Rules:
- PRIMARY SOURCES: Use the provided video transcripts, PDF text, and visual descriptions as your main sources.
- CITATIONS: Always reference the source (video name with timestamp OR PDF name with chunk/page).
- VISUAL INTEGRATION: When diagrams, charts, or screenshots are relevant, explicitly mention what they show.
- TRANSPARENCY: If the sources don't fully answer the question, state this clearly.
- STYLE: Clear, student-friendly explanations with proper structure.
- COMPREHENSIVENESS: Combine textual explanations with visual information for complete understanding.
- CROSS-REFERENCE: When videos and PDFs discuss the same topic, connect the information.
"""
    
    user_prompt = f"""Question: {query}

Sources (Videos + PDFs, Text + Visual):
{context}

Provide a clear, educational answer based on all available sources above."""
    
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response["message"]["content"]
        
    except Exception as e:
        return f"❌ Error generating response: {e}"


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def print_header():
    """Print welcome header."""
    print("\n" + "="*70)
    print("  🤖 UNIFIED RAG TEACHING ASSISTANT")
    print("="*70)


def print_stats(counts: Dict[str, int]):
    """Print database statistics."""
    print("\n📊 Available Content:")
    
    if counts['video_text'] > 0 or counts['video_frames'] > 0:
        print(f"   🎥 Videos: {counts['video_text']} text chunks, {counts['video_frames']} frames")
    
    if counts['pdf_text'] > 0 or counts['pdf_images'] > 0:
        print(f"   📄 PDFs: {counts['pdf_text']} text chunks, {counts['pdf_images']} images")
    
    total = sum(counts.values())
    print(f"   📚 Total: {total} searchable items")


def print_sources(results: Dict[str, List]):
    """Print which sources were used."""
    sources = []
    
    if results["video_text"]:
        video_names = set(r["metadata"].get("video", "") for r in results["video_text"])
        sources.append(f"{len(video_names)} video(s)")
    
    if results["pdf_text"]:
        pdf_names = set(r["metadata"].get("document", "") for r in results["pdf_text"])
        sources.append(f"{len(pdf_names)} PDF(s)")
    
    if results["video_frames"]:
        sources.append(f"{len(results['video_frames'])} video frame(s)")
    
    if results["pdf_images"]:
        sources.append(f"{len(results['pdf_images'])} PDF image(s)")
    
    if sources:
        print(f"\n📚 Sources used: {', '.join(sources)}")


# ============================================================================
# CONVERSATION LOOP
# ============================================================================

def conversation_loop(collections: Dict, counts: Dict):
    """Main conversational interface."""
    print_header()
    print_stats(counts)
    
    print("\n" + "="*70)
    print("  💡 Ask me anything about your videos and PDFs!")
    print("  Type 'exit' or 'quit' to end the session")
    print("="*70 + "\n")
    
    conversation_count = 0
    
    while True:
        # Get user query
        try:
            query = input("❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break
        
        # Check for exit
        if query.lower() in ['exit', 'quit', 'bye', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Skip empty queries
        if not query:
            continue
        
        conversation_count += 1
        
        print(f"\n🔍 Searching knowledge base...")
        
        # Retrieve content
        results = retrieve_all_content(query, collections)
        
        # Check if any results found
        total_results = sum(len(results[key]) for key in results)
        
        if total_results == 0:
            print("❌ No relevant content found.")
            print("💡 Try rephrasing your question or using different keywords\n")
            continue
        
        # Show sources
        print_sources(results)
        
        # Build context
        context = build_context(results)
        
        # Generate response
        print("\n💭 Generating explanation...\n")
        print("─" * 70)
        
        response = teach(query, context)
        print(response)
        
        print("─" * 70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    try:
        # Initialize database
        collections, counts = init_db()
        
        # Start conversation loop
        conversation_loop(collections, counts)
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()