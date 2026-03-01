import json
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
import ollama

# CONFIG
PDF_CHUNKS_DIR = Path("pdf_chunks")
PDF_IMAGES_DIR = Path("pdf_images")
DB_DIR = Path("vector_db")
PDF_TEXT_COLLECTION = "pdf_chunks"
PDF_IMAGE_COLLECTION = "pdf_images"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 50

DB_DIR.mkdir(exist_ok=True)

# Initialize ChromaDB
client = chromadb.PersistentClient(
    path=str(DB_DIR),
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

text_collection = client.get_or_create_collection(
    name=PDF_TEXT_COLLECTION,
    metadata={"hnsw:space": "cosine"},
    embedding_function=None
)

image_collection = client.get_or_create_collection(
    name=PDF_IMAGE_COLLECTION,
    metadata={"hnsw:space": "cosine"},
    embedding_function=None
)


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


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts."""
    return [embed_text(text) for text in texts]


def get_existing_ids(collection) -> set:
    """Get existing IDs from collection."""
    try:
        result = collection.get()
        return set(result["ids"])
    except:
        return set()


def process_text_chunks(doc_name: str, existing_ids: set) -> int:
    """Process PDF text chunks."""
    chunks_file = PDF_CHUNKS_DIR / f"{doc_name}_chunks.json"
    
    if not chunks_file.exists():
        return 0
    
    with open(chunks_file, encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Filter new chunks
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    
    if not new_chunks:
        print(f"⏭️  Text: {doc_name} ({len(chunks)} chunks already embedded)")
        return 0
    
    print(f"📝 Embedding {len(new_chunks)} text chunks from {doc_name}...")
    
    # Process in batches
    total_added = 0
    
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i:i + BATCH_SIZE]
        
        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [
            {
                "document": c["document"],
                "chunk_index": c["chunk_index"],
                "word_count": c["word_count"],
                "source_type": "pdf"
            }
            for c in batch
        ]
        
        embeddings = embed_batch(texts)
        
        text_collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        total_added += len(batch)
        print(f"  Progress: {total_added}/{len(new_chunks)}")
    
    print(f"✅ Text: {doc_name} - {total_added} new chunks")
    return total_added


def describe_image(image_path: Path, vision_model: str) -> str:
    """Describe PDF image using vision model."""
    try:
        response = ollama.chat(
            model=vision_model,
            messages=[{
                'role': 'user',
                'content': 'Describe this image from a PDF document. Focus on: diagrams, charts, tables, text, formulas, or any visual information.',
                'images': [str(image_path)]
            }]
        )
        return response['message']['content']
    except Exception as e:
        print(f"  ⚠️  Vision error: {e}")
        return ""


def process_images(doc_name: str, existing_ids: set, vision_model: str) -> int:
    """Process PDF images."""
    images_file = PDF_CHUNKS_DIR / f"{doc_name}_images.json"
    
    if not images_file.exists():
        return 0
    
    with open(images_file, encoding='utf-8') as f:
        images = json.load(f)
    
    if not images:
        return 0
    
    # Filter new images
    new_images = [
        img for img in images
        if f"{img['document']}_img_p{img['page']}_{img['image_index']}" not in existing_ids
    ]
    
    if not new_images:
        print(f"⏭️  Images: {doc_name} ({len(images)} images already embedded)")
        return 0
    
    print(f"🖼️  Embedding {len(new_images)} images from {doc_name}...")
    
    embedded_images = []
    
    for img in new_images:
        img_path = Path(img["path"])
        
        if not img_path.exists():
            continue
        
        description = describe_image(img_path, vision_model)
        
        if not description:
            continue
        
        embedding = embed_text(description)
        
        img_id = f"{img['document']}_img_p{img['page']}_{img['image_index']}"
        
        embedded_images.append({
            "id": img_id,
            "document": description,
            "embedding": embedding,
            "metadata": {
                "document": img["document"],
                "page": img["page"],
                "image_index": img["image_index"],
                "path": str(img_path),
                "source_type": "pdf_image"
            }
        })
    
    # Store in ChromaDB
    if embedded_images:
        image_collection.add(
            ids=[img["id"] for img in embedded_images],
            documents=[img["document"] for img in embedded_images],
            embeddings=[img["embedding"] for img in embedded_images],
            metadatas=[img["metadata"] for img in embedded_images]
        )
    
    print(f"✅ Images: {doc_name} - {len(embedded_images)} new images")
    return len(embedded_images)


def find_vision_model() -> str:
    """Find available vision model (same as embed_frames.py)."""
    models = ["moondream", "llava", "bakllava"]
    
    for model in models:
        try:
            ollama.chat(model=model, messages=[{'role': 'user', 'content': 'hi'}])
            return model
        except:
            continue
    
    return None


def main():
    print("="*70)
    print("  PDF EMBEDDING GENERATION")
    print("="*70)
    print()
    
    # Find vision model
    vision_model = find_vision_model()
    if vision_model:
        print(f"🤖 Vision model: {vision_model}\n")
    else:
        print("⚠️  No vision model - skipping image embedding\n")
    
    # Get existing IDs
    text_existing = get_existing_ids(text_collection)
    image_existing = get_existing_ids(image_collection)
    
    print(f"📊 Existing: {len(text_existing)} text chunks, {len(image_existing)} images\n")
    
    # Get all PDF chunk files
    chunk_files = list(PDF_CHUNKS_DIR.glob("*_chunks.json"))
    
    if not chunk_files:
        print("❌ No PDF chunks found")
        print("💡 Run pdf_to_chunks.py first")
        return
    
    doc_names = [f.stem.replace("_chunks", "") for f in chunk_files]
    
    print(f"📁 Found {len(doc_names)} PDF(s)\n")
    
    total_text = 0
    total_images = 0
    
    for doc_name in doc_names:
        # Embed text
        text_added = process_text_chunks(doc_name, text_existing)
        total_text += text_added
        
        # Embed images (if vision model available)
        if vision_model:
            images_added = process_images(doc_name, image_existing, vision_model)
            total_images += images_added
        
        print()
    
    print("="*70)
    print(f"🎉 PDF Embedding Complete")
    print(f"📊 Total: {text_collection.count()} text chunks, {image_collection.count()} images")
    print(f"➕ New: {total_text} text chunks, {total_images} images")
    print("="*70)


if __name__ == "__main__":
    main()