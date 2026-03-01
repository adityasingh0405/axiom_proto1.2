"""
Optimized JSON to Chunks with Configurable Profiles
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

# ============================================================================
# CHUNKING PROFILES
# ============================================================================

CHUNKING_PROFILES = {
    "podcast": {
        "max_words": 400,
        "max_duration": 60,
        "overlap_words": 50,
        "description": "Long conversational content",
        "best_for": "Interviews, discussions, debates"
    },
    "tutorial": {
        "max_words": 150,
        "max_duration": 15,
        "overlap_words": 10,
        "description": "Step-by-step instructions",
        "best_for": "How-to guides, quick tips"
    },
    "lecture": {
        "max_words": 300,
        "max_duration": 45,
        "overlap_words": 30,
        "description": "Educational content",
        "best_for": "Classes, presentations, workshops"
    },
    "coding": {
        "max_words": 250,
        "max_duration": 90,
        "overlap_words": 40,
        "description": "Programming tutorials",
        "best_for": "Code walkthroughs, live coding"
    },
    "news": {
        "max_words": 180,
        "max_duration": 20,
        "overlap_words": 15,
        "description": "News segments",
        "best_for": "News clips, announcements"
    }
}

# CONFIG
JSON_DIR = Path("json")
CHUNKS_DIR = Path("chunks")
MAX_WORKERS = 4  # CPU-bound task, more workers = faster

CHUNKS_DIR.mkdir(exist_ok=True)


# ============================================================================
# PROFILE SELECTION
# ============================================================================

def display_chunking_profiles():
    """Display available chunking profiles."""
    print("\n" + "="*70)
    print("  CHUNKING PROFILE OPTIONS")
    print("="*70 + "\n")
    
    for i, (key, profile) in enumerate(CHUNKING_PROFILES.items(), 1):
        print(f"{i}. [{key.upper()}]")
        print(f"   Words: {profile['max_words']} | Duration: {profile['max_duration']}s | Overlap: {profile['overlap_words']}")
        print(f"   → {profile['description']}")
        print(f"   💡 Best for: {profile['best_for']}\n")


def select_chunking_profile() -> dict:
    """Let user select chunking profile."""
    display_chunking_profiles()
    
    keys = list(CHUNKING_PROFILES.keys())
    default_key = "lecture"
    
    print(f"Choose chunking profile (1-{len(keys)}) or press Enter for default [{default_key}]: ", end="")
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
    
    selected_profile = CHUNKING_PROFILES[selected_key]
    
    print(f"\n✅ Selected: {selected_key.upper()}")
    print(f"   Max Words: {selected_profile['max_words']} | Max Duration: {selected_profile['max_duration']}s")
    print(f"   Overlap: {selected_profile['overlap_words']} words")
    print("="*70 + "\n")
    
    return selected_profile


# ============================================================================
# CHUNKING LOGIC
# ============================================================================

def chunk_transcript(json_path: Path, profile: dict) -> tuple[Path, int]:
    """
    Chunk a transcript with configurable profile.
    Returns (json_path, num_chunks).
    """
    video_name = json_path.stem
    output_path = CHUNKS_DIR / f"{video_name}_chunks.json"
    
    # Validate existing chunks
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
                if existing and isinstance(existing, list):
                    print(f"⏭️  Skip: {video_name} ({len(existing)} chunks)")
                    return (json_path, len(existing))
        except (json.JSONDecodeError, KeyError):
            print(f"⚠️  Re-chunking corrupted: {video_name}")
    
    print(f"🧩 Chunking: {video_name}")
    
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON: {video_name}")
        return (json_path, 0)
    
    segments = data.get("segments", [])
    if not segments:
        print(f"⚠️  Empty transcript: {video_name}")
        return (json_path, 0)
    
    # Extract profile settings
    MAX_WORDS = profile["max_words"]
    MAX_DURATION = profile["max_duration"]
    OVERLAP_WORDS = profile["overlap_words"]
    
    chunks = []
    buffer = []  # Segment buffer for overlap
    chunk_index = 0
    
    total_words = 0
    start_time = segments[0]["start"]
    end_time = segments[0]["end"]
    
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        
        seg_words = text.split()
        seg_word_count = len(seg_words)
        
        buffer.append({
            "text": text,
            "words": seg_words,
            "word_count": seg_word_count,
            "start": seg["start"],
            "end": seg["end"]
        })
        
        total_words += seg_word_count
        end_time = seg["end"]
        duration = end_time - start_time
        
        # Create chunk if limits reached
        if total_words >= MAX_WORDS or duration >= MAX_DURATION:
            chunk_text = " ".join(s["text"] for s in buffer)
            
            # Generate stable hash-based ID
            chunk_hash = hashlib.md5(
                f"{video_name}_{chunk_index}_{start_time}".encode()
            ).hexdigest()[:12]
            
            chunks.append({
                "chunk_id": f"{video_name}_{chunk_index}_{chunk_hash}",
                "video": video_name,
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "duration": round(duration, 2),
                "text": chunk_text,
                "word_count": total_words
            })
            
            chunk_index += 1
            
            # Keep overlap: last N words for context
            overlap_buffer = []
            overlap_words = 0
            
            for s in reversed(buffer):
                if overlap_words + s["word_count"] <= OVERLAP_WORDS:
                    overlap_buffer.insert(0, s)
                    overlap_words += s["word_count"]
                else:
                    break
            
            buffer = overlap_buffer
            total_words = sum(s["word_count"] for s in buffer)
            start_time = buffer[0]["start"] if buffer else seg["end"]
    
    # Final chunk
    if buffer:
        chunk_text = " ".join(s["text"] for s in buffer)
        chunk_hash = hashlib.md5(
            f"{video_name}_{chunk_index}_{start_time}".encode()
        ).hexdigest()[:12]
        
        chunks.append({
            "chunk_id": f"{video_name}_{chunk_index}_{chunk_hash}",
            "video": video_name,
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "duration": round(end_time - start_time, 2),
            "text": chunk_text,
            "word_count": total_words
        })
    
    # Write chunks atomically
    output_path.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    print(f"✅ {video_name}: {len(chunks)} chunks")
    return (json_path, len(chunks))


# ============================================================================
# FILE HANDLING
# ============================================================================

def get_json_files() -> List[Path]:
    """Get all JSON transcript files sorted by size."""
    if not JSON_DIR.exists():
        return []
    files = list(JSON_DIR.glob("*.json"))
    return sorted(files, key=lambda x: x.stat().st_size)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("  🧩 JSON TO CHUNKS - CONFIGURABLE CHUNKING")
    print("="*70)
    
    # Select chunking profile
    profile = select_chunking_profile()
    
    # Get JSON files
    json_files = get_json_files()
    
    if not json_files:
        print("❌ No JSON files found")
        return
    
    print(f"📁 Found {len(json_files)} file(s)\n")
    
    total_chunks = 0
    
    # Parallel processing
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(chunk_transcript, jf, profile): jf for jf in json_files}
        
        for future in as_completed(futures):
            _, num_chunks = future.result()
            total_chunks += num_chunks
    
    print(f"\n{'='*70}")
    print(f"🎉 Chunking Complete")
    print(f"📊 Total Chunks: {total_chunks} from {len(json_files)} file(s)")
    print(f"🔧 Profile: {profile['max_words']} words, {profile['max_duration']}s, {profile['overlap_words']} overlap")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()