"""
embed_frames.py - FIXED model detection
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
import ollama

# CONFIG
FRAMES_DIR = Path("frames")
DB_DIR = Path("vector_db")
COLLECTION_NAME = "video_frames"
EMBED_MODEL = "nomic-embed-text"

# Try multiple vision models in order of preference
VISION_MODELS = [
    "moondream",
    "llava:7b",
    "llava",
    "bakllava",
    "llava-phi3"
]

DB_DIR.mkdir(exist_ok=True)


def find_available_vision_model() -> str:
    """Find first available vision model."""
    print("🔍 Checking available vision models...")
    
    try:
        # Get list of installed models
        result = ollama.list()
        
        # Handle different response formats
        if isinstance(result, dict):
            models_list = result.get('models', [])
        else:
            models_list = result
        
        installed = []
        for model in models_list:
            if isinstance(model, dict):
                # Try different key names
                name = model.get('name') or model.get('model') or model.get('id')
                if name:
                    installed.append(name)
            elif isinstance(model, str):
                installed.append(model)
        
        print(f"   Found {len(installed)} installed model(s)")
        
        # Check each preferred model
        for pref_model in VISION_MODELS:
            for installed_model in installed:
                # Match full name or partial name
                if pref_model in installed_model or installed_model.startswith(pref_model):
                    print(f"✅ Found vision model: {installed_model}")
                    return installed_model
        
        print("⚠️  No vision models found!")
        if installed:
            print("\n📥 Available models on your system:")
            for m in installed:
                print(f"   - {m}")
        
        return None
        
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        print("\n💡 Trying direct test instead...")
        
        # Fallback: try each model directly
        for model in VISION_MODELS:
            try:
                print(f"   Testing {model}...", end=" ")
                # Quick test
                ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': 'test'}]
                )
                print("✅")
                return model
            except:
                print("❌")
        
        return None


def download_vision_model() -> str:
    """Try to download a vision model."""
    print("\n📥 Attempting to download vision model...")
    
    for model in VISION_MODELS:
        print(f"   Trying {model}...")
        try:
            ollama.pull(model)
            print(f"   ✅ Successfully downloaded {model}")
            return model
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    return None


# Initialize ChromaDB
client = chromadb.PersistentClient(
    path=str(DB_DIR),
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
    embedding_function=None
)


def describe_image(image_path: Path, vision_model: str) -> str:
    """Generate description of image using vision model."""
    try:
        response = ollama.chat(
            model=vision_model,
            messages=[{
                'role': 'user',
                'content': 'Describe what you see in this image in 2-3 sentences. Focus on: text/code visible, main objects, actions, and UI elements.',
                'images': [str(image_path)]
            }]
        )
        return response['message']['content']
    except Exception as e:
        print(f"  ⚠️  Vision error: {e}")
        return ""


def embed_text(text: str) -> List[float]:
    """Generate embedding from text."""
    try:
        response = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=text
        )
        return response["embedding"]
    except Exception as e:
        print(f"  ⚠️  Embedding error: {e}")
        return [0.0] * 768


def process_frame(frame_meta: Dict, vision_model: str) -> Dict:
    """Process single frame: describe + embed."""
    frame_path = Path(frame_meta["path"])
    
    if not frame_path.exists():
        return None
    
    # Generate description using vision model
    description = describe_image(frame_path, vision_model)
    
    if not description:
        return None
    
    # Generate embedding from description
    embedding = embed_text(description)
    
    return {
        "frame_id": f"{frame_meta['video']}_frame_{frame_meta['frame_index']}",
        "video": frame_meta["video"],
        "frame_index": frame_meta["frame_index"],
        "timestamp": frame_meta["timestamp"],
        "path": str(frame_path),
        "description": description,
        "embedding": embedding
    }


def get_existing_ids() -> set:
    """Get all existing frame IDs."""
    try:
        result = collection.get()
        return set(result["ids"])
    except:
        return set()


def process_video_frames(video_name: str, existing_ids: set, vision_model: str) -> Tuple[str, int]:
    """Process all frames for a video."""
    frame_dir = FRAMES_DIR / video_name
    metadata_file = frame_dir / f"{video_name}_frames_metadata.json"
    
    if not metadata_file.exists():
        print(f"⚠️  No metadata for {video_name}")
        return (video_name, 0)
    
    with open(metadata_file, encoding='utf-8') as f:
        frames_metadata = json.load(f)
    
    # Filter out already processed frames
    new_frames = [
        f for f in frames_metadata 
        if f"{f['video']}_frame_{f['frame_index']}" not in existing_ids
    ]
    
    if not new_frames:
        print(f"⏭️  Skip: {video_name} ({len(frames_metadata)} frames already embedded)")
        return (video_name, 0)
    
    print(f"🖼️  Processing {len(new_frames)} new frames from {video_name}")
    
    embedded_frames = []
    
    for i, frame_meta in enumerate(new_frames, 1):
        print(f"  [{i}/{len(new_frames)}] Frame {frame_meta['frame_index']}... ", end="", flush=True)
        
        result = process_frame(frame_meta, vision_model)
        
        if result:
            embedded_frames.append(result)
            print("✅")
        else:
            print("❌")
    
    # Store in ChromaDB
    if embedded_frames:
        ids = [f["frame_id"] for f in embedded_frames]
        documents = [f["description"] for f in embedded_frames]
        embeddings = [f["embedding"] for f in embedded_frames]
        metadatas = [
            {
                "video": f["video"],
                "frame_index": f["frame_index"],
                "timestamp": f["timestamp"],
                "path": f["path"]
            }
            for f in embedded_frames
        ]
        
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    return (video_name, len(embedded_frames))


def main():
    print("="*70)
    print("  FRAME EMBEDDING GENERATION")
    print("="*70)
    print()
    
    # Find or download vision model
    vision_model = find_available_vision_model()
    
    if not vision_model:
        print("\n⚠️  No vision model available!")
        print("\n💡 Let's try using moondream directly...")
        
        # Direct test for moondream
        try:
            print("   Testing moondream...", end=" ")
            ollama.chat(
                model="moondream",
                messages=[{'role': 'user', 'content': 'hi'}]
            )
            print("✅ Works!")
            vision_model = "moondream"
        except Exception as e:
            print(f"❌ Failed: {e}")
            
            choice = input("\nTry to download moondream? (y/n): ").strip().lower()
            
            if choice == 'y':
                vision_model = download_vision_model()
            
            if not vision_model:
                print("\n❌ Cannot proceed without vision model")
                print("\n💡 Manual check:")
                print("   1. Run: ollama list")
                print("   2. Verify moondream is listed")
                print("   3. Run: ollama run moondream (then type /bye to exit)")
                return
    
    print(f"\n🤖 Using vision model: {vision_model}\n")
    
    if not FRAMES_DIR.exists():
        print("❌ Frames directory not found")
        print("💡 Run extract_frames.py first")
        return
    
    # Get all video frame directories
    video_dirs = [d for d in FRAMES_DIR.iterdir() if d.is_dir()]
    
    if not video_dirs:
        print("❌ No frame directories found")
        return
    
    print(f"📁 Found {len(video_dirs)} video(s) with frames")
    print(f"🔍 Checking existing embeddings...\n")
    
    existing_ids = get_existing_ids()
    print(f"📊 {len(existing_ids)} frames already in DB\n")
    
    total_embedded = 0
    
    for video_dir in video_dirs:
        video_name = video_dir.name
        _, num_embedded = process_video_frames(video_name, existing_ids, vision_model)
        total_embedded += num_embedded
    
    # Get final stats
    final_count = collection.count()
    
    print()
    print("="*70)
    print(f"🎉 Embedding Complete")
    print(f"📊 Total Frames in DB: {final_count}")
    print(f"➕ New Frames Added: {total_embedded}")
    print("="*70)


if __name__ == "__main__":
    main()