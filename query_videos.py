"""
Multi-Modal RAG Query: Text + Visual Context
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import chromadb
from chromadb.config import Settings
import ollama

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "vector_db"
TEXT_COLLECTION = "video_chunks"
FRAME_COLLECTION = "video_frames"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"

# MODE OPTIONS: "locate", "teach", "notes"
MODE = "notes"

# Retrieval settings
TOP_K_TEXT = 8
TOP_K_FRAMES = 3
PER_VIDEO = 2
MAX_DISTANCE = 260


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def init_db():
    """Initialize ChromaDB connections."""
    if not DB_DIR.exists():
        print(f"❌ Database not found at: {DB_DIR}")
        exit(1)
    
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Text collection
    text_collection = client.get_or_create_collection(
        name=TEXT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None
    )
    
    # Frame collection
    try:
        frame_collection = client.get_or_create_collection(
            name=FRAME_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )
        has_frames = frame_collection.count() > 0
    except:
        frame_collection = None
        has_frames = False
    
    text_count = text_collection.count()
    frame_count = frame_collection.count() if frame_collection else 0
    
    print(f"📊 Database: {text_count} text chunks, {frame_count} frames")
    
    if text_count == 0:
        print("⚠️  No text chunks. Run chunk_embeddings.py first.")
        exit(1)
    
    return text_collection, frame_collection, has_frames


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
        exit(1)


def confidence_label(distance: float) -> str:
    """Convert distance to confidence label."""
    if distance < 220:
        return "Main explanation"
    elif distance < 245:
        return "Supporting mention"
    else:
        return "Weak mention"


def retrieve_text_chunks(
    collection,
    query: str,
    top_k: int = TOP_K_TEXT,
    per_video: int = PER_VIDEO,
    max_distance: float = MAX_DISTANCE
) -> Dict[str, List[Dict[str, Any]]]:
    """Smart retrieval: groups by video and limits chunks per video."""
    query_embedding = embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    if not results["ids"][0]:
        return {}
    
    grouped = {}
    
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if dist > max_distance:
            continue
        
        video = meta["video"]
        grouped.setdefault(video, []).append({
            "type": "text",
            "text": doc,
            "start": meta["start"],
            "end": meta["end"],
            "duration": meta.get("duration", meta["end"] - meta["start"]),
            "distance": dist,
            "confidence": confidence_label(dist)
        })
    
    # Sort by relevance and keep best chunks per video
    for video in grouped:
        grouped[video] = sorted(
            grouped[video],
            key=lambda x: x["distance"]
        )[:per_video]
    
    return grouped


def retrieve_frames(
    collection,
    query: str,
    top_k: int = TOP_K_FRAMES
) -> List[Dict[str, Any]]:
    """Search video frames."""
    if not collection:
        return []
    
    query_embedding = embed_query(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    if not results["ids"][0]:
        return []
    
    frames = []
    for i in range(len(results["ids"][0])):
        frames.append({
            "type": "frame",
            "video": results["metadatas"][0][i]["video"],
            "timestamp": results["metadatas"][0][i]["timestamp"],
            "frame_index": results["metadatas"][0][i]["frame_index"],
            "description": results["documents"][0][i],
            "path": results["metadatas"][0][i]["path"],
            "distance": results["distances"][0][i] if "distances" in results else None
        })
    
    return frames


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS."""
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


# ============================================================================
# MODE 1: LOCATE - Find timestamps
# ============================================================================

def locate_mode(text_chunks: Dict[str, List[Dict]], frames: List[Dict]):
    """Display locations with both text and frames."""
    print("\n" + "="*70)
    print("📍 TOPIC LOCATIONS")
    print("="*70 + "\n")
    
    # Combine and sort by video + timestamp
    all_results = []
    
    for video, chunks in text_chunks.items():
        for chunk in chunks:
            all_results.append({
                "video": video,
                "timestamp": chunk["start"],
                "type": "text",
                "data": chunk
            })
    
    for frame in frames:
        all_results.append({
            "video": frame["video"],
            "timestamp": frame["timestamp"],
            "type": "frame",
            "data": frame
        })
    
    all_results.sort(key=lambda x: (x["video"], x["timestamp"]))
    
    current_video = None
    for result in all_results:
        if result["video"] != current_video:
            if current_video is not None:
                print()
            print(f"🎥 Video: {result['video']}\n")
            current_video = result["video"]
        
        if result["type"] == "text":
            data = result["data"]
            start = format_timestamp(data["start"])
            end = format_timestamp(data["end"])
            print(f"  💬 {start} – {end}")
            print(f"     📊 {data['confidence']}")
            print(f"     {data['text'][:100]}...")
            print()
        else:
            data = result["data"]
            ts = format_timestamp(data["timestamp"])
            print(f"  🖼️  {ts} (Frame #{data['frame_index']})")
            print(f"     {data['description'][:100]}...")
            print(f"     📁 {data['path']}")
            print()


# ============================================================================
# MODE 2: TEACH - Interactive teaching
# ============================================================================

def teach_mode(query: str, text_chunks: Dict[str, List[Dict]], frames: List[Dict]):
    """Generate explanation using both text and visual context."""
    print("\n" + "="*70)
    print("🧠 TEACHING MODE")
    print("="*70 + "\n")
    
    # Build context
    context_parts = []
    
    context_parts.append("=== TEXT TRANSCRIPT CONTEXT ===\n")
    for video, chunks in text_chunks.items():
        for chunk in chunks:
            ts = format_timestamp(chunk["start"])
            context_parts.append(
                f"[Text] {video} @ {ts}\n{chunk['text']}\n"
            )
    
    if frames:
        context_parts.append("\n=== VISUAL CONTEXT (What's shown on screen) ===\n")
        for frame in frames:
            ts = format_timestamp(frame["timestamp"])
            context_parts.append(
                f"[Visual] {frame['video']} @ {ts}\n{frame['description']}\n"
            )
    
    context = "\n".join(context_parts)
    
    system_prompt = """You are a precise AI Teaching Assistant with access to both audio transcripts AND visual information (screenshots) from educational videos.

Rules:
- PRIMARY SOURCES: Use both transcript text AND visual descriptions as your main sources.
- TIMESTAMPS: Always include timestamps [MM:SS] when referencing content.
- VISUAL REFERENCES: When visual context is relevant, explicitly mention what was shown on screen.
- TRANSPARENCY: If the sources don't fully answer the question, state this clearly.
- STYLE: Clear headings, bullet points, student-friendly explanations.
- INTEGRATION: Combine what was SAID (transcript) with what was SHOWN (visuals) for complete understanding.
"""
    
    user_prompt = f"""Question: {query}

Sources (Audio + Visual):
{context}

Provide a comprehensive answer using both the transcript and visual information."""
    
    print("💭 Generating explanation...\n")
    
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        print(response["message"]["content"])
        
    except Exception as e:
        print(f"❌ LLM error: {e}")


# ============================================================================
# MODE 3: NOTES - Generate structured notes
# ============================================================================

def notes_mode(query: str, text_chunks: Dict[str, List[Dict]], frames: List[Dict]):
    """Generate structured notes with visual references."""
    print("\n" + "="*70)
    print("📝 NOTES GENERATION")
    print("="*70 + "\n")
    
    # Build context
    context_parts = []
    
    context_parts.append("=== TEXT TRANSCRIPT ===\n")
    for video, chunks in text_chunks.items():
        for chunk in chunks:
            ts = format_timestamp(chunk["start"])
            context_parts.append(f"[{video} @ {ts}]\n{chunk['text']}\n")
    
    if frames:
        context_parts.append("\n=== VISUAL CONTENT ===\n")
        for frame in frames:
            ts = format_timestamp(frame["timestamp"])
            context_parts.append(f"[{frame['video']} @ {ts}]\n{frame['description']}\n")
    
    context = "\n".join(context_parts)
    
    system_prompt = """You are an expert note-taker with access to both audio and visual content from videos.

Create comprehensive, well-structured notes that:
- Use clear hierarchical organization (headers, subheadings, bullets)
- Extract key concepts with definitions
- Note important visual elements (diagrams, code shown, UI elements)
- Include timestamps [MM:SS] for all major points
- Combine audio explanations with visual demonstrations
- Use markdown formatting
- Distinguish between what was SAID vs what was SHOWN

Format:
# Main Topic

## Key Concepts
- Concept 1 [MM:SS] (explained in audio)
- Visual: What was shown at [MM:SS]

## Detailed Explanation
...

## Visual Examples
- Screenshot at [MM:SS]: Description
"""
    
    user_prompt = f"""Topic: {query}

Content (Audio + Visual):
{context}

Generate comprehensive notes covering all key information."""
    
    print("📝 Generating notes...\n")
    
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        notes = response["message"]["content"]
        print(notes)
        
        # Save notes
        notes_dir = BASE_DIR / "generated_notes"
        notes_dir.mkdir(exist_ok=True)
        notes_file = notes_dir / f"{query[:50].replace(' ', '_')}.md"
        notes_file.write_text(notes, encoding="utf-8")
        
        print(f"\n💾 Notes saved to: {notes_file}")
        
    except Exception as e:
        print(f"❌ LLM error: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print(f"🤖 MULTI-MODAL RAG QUERY - Mode: {MODE.upper()}")
    print("="*70 + "\n")
    
    # Initialize
    text_collection, frame_collection, has_frames = init_db()
    
    if not has_frames:
        print("⚠️  No frame embeddings found. Visual context disabled.")
        print("💡 Run extract_frames.py and embed_frames.py for multi-modal search\n")
    
    # Get query
    query = input("Enter your query: ").strip()
    
    if not query:
        print("❌ Query cannot be empty")
        return
    
    print()
    print("🔍 Searching...")
    
    # Search both text and frames
    text_chunks = retrieve_text_chunks(text_collection, query)
    frames = retrieve_frames(frame_collection, query) if has_frames else []
    
    if not text_chunks and not frames:
        print("❌ No relevant content found.")
        print("💡 Try:")
        print("  - Broadening your query")
        print("  - Using different keywords")
        print("  - Lowering MAX_DISTANCE threshold")
        return
    
    total_text = sum(len(chunks) for chunks in text_chunks.values())
    print(f"✅ Found {total_text} text chunks, {len(frames)} frames\n")
    
    # Execute based on mode
    if MODE == "locate":
        locate_mode(text_chunks, frames)
    elif MODE == "teach":
        teach_mode(query, text_chunks, frames)
    elif MODE == "notes":
        notes_mode(query, text_chunks, frames)
    else:
        print(f"❌ Unknown mode: {MODE}")
        print("Valid modes: locate, teach, notes")


if __name__ == "__main__":
    main()