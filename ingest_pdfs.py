import json
from pathlib import Path
from typing import List, Dict, Tuple
import fitz  
import hashlib
from PIL import Image
import io

# CONFIG
PDF_DIR = Path("pdfs")
PDF_CHUNKS_DIR = Path("pdf_chunks")
PDF_IMAGES_DIR = Path("pdf_images")

# Chunking settings (similar to video chunks)
CHUNK_MAX_WORDS = 300
CHUNK_OVERLAP_WORDS = 50

PDF_DIR.mkdir(exist_ok=True)
PDF_CHUNKS_DIR.mkdir(exist_ok=True)
PDF_IMAGES_DIR.mkdir(exist_ok=True)


def extract_text_and_images(pdf_path: Path) -> Tuple[str, List[Dict]]:
    """
    Extract text and images from PDF.
    Returns (full_text, images_metadata).
    """
    doc_name = pdf_path.stem
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Cannot open PDF: {e}")
        return "", []
    
    full_text = []
    images_metadata = []
    
    print(f"📄 Extracting from {doc_name} ({len(doc)} pages)")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text
        text = page.get_text()
        if text.strip():
            full_text.append(f"[Page {page_num + 1}]\n{text}")
        
        # Extract images
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save image
                img_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                img_filename = f"{doc_name}_p{page_num + 1}_img{img_index}_{img_hash}.png"
                img_dir = PDF_IMAGES_DIR / doc_name
                img_dir.mkdir(exist_ok=True)
                img_path = img_dir / img_filename
                
                # Convert and save
                image = Image.open(io.BytesIO(image_bytes))
                image.save(img_path, "PNG")
                
                images_metadata.append({
                    "document": doc_name,
                    "page": page_num + 1,
                    "image_index": img_index,
                    "path": str(img_path)
                })
                
            except Exception as e:
                print(f"  ⚠️  Failed to extract image on page {page_num + 1}: {e}")
    
    doc.close()
    
    full_text_str = "\n\n".join(full_text)
    
    print(f"  ✅ Extracted {len(full_text_str.split())} words, {len(images_metadata)} images")
    
    return full_text_str, images_metadata


def chunk_text(text: str, doc_name: str) -> List[Dict]:
    """Chunk PDF text with overlap."""
    words = text.split()
    
    if len(words) == 0:
        return []
    
    chunks = []
    chunk_index = 0
    i = 0
    
    while i < len(words):
        # Take chunk_size words
        chunk_words = words[i:i + CHUNK_MAX_WORDS]
        chunk_text = " ".join(chunk_words)
        
        # Generate chunk ID
        chunk_hash = hashlib.md5(
            f"{doc_name}_{chunk_index}_{chunk_text[:50]}".encode()
        ).hexdigest()[:12]
        
        chunks.append({
            "chunk_id": f"{doc_name}_chunk_{chunk_index}_{chunk_hash}",
            "document": doc_name,
            "chunk_index": chunk_index,
            "text": chunk_text,
            "word_count": len(chunk_words),
            "source_type": "pdf"
        })
        
        chunk_index += 1
        
        # Move forward with overlap
        i += CHUNK_MAX_WORDS - CHUNK_OVERLAP_WORDS
    
    return chunks


def process_pdf(pdf_path: Path) -> Tuple[Path, int, int]:
    """
    Process single PDF.
    Returns (pdf_path, num_chunks, num_images).
    """
    doc_name = pdf_path.stem
    output_chunks = PDF_CHUNKS_DIR / f"{doc_name}_chunks.json"
    output_images_meta = PDF_CHUNKS_DIR / f"{doc_name}_images.json"
    
    # Check if already processed
    if output_chunks.exists() and output_images_meta.exists():
        try:
            with open(output_chunks) as f:
                chunks = json.load(f)
            with open(output_images_meta) as f:
                images = json.load(f)
            print(f"⏭️  Skip: {doc_name} ({len(chunks)} chunks, {len(images)} images)")
            return (pdf_path, len(chunks), len(images))
        except:
            print(f"⚠️  Re-processing corrupt data: {doc_name}")
    
    # Extract text and images
    full_text, images_metadata = extract_text_and_images(pdf_path)
    
    if not full_text:
        print(f"⚠️  No text extracted from {doc_name}")
        return (pdf_path, 0, 0)
    
    # Chunk text
    chunks = chunk_text(full_text, doc_name)
    
    # Save chunks
    with open(output_chunks, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    # Save image metadata
    with open(output_images_meta, 'w', encoding='utf-8') as f:
        json.dump(images_metadata, f, indent=2)
    
    print(f"✅ {doc_name}: {len(chunks)} chunks, {len(images_metadata)} images")
    
    return (pdf_path, len(chunks), len(images_metadata))


def get_pdfs() -> List[Path]:
    """Get all PDF files."""
    if not PDF_DIR.exists():
        return []
    return sorted(list(PDF_DIR.glob("*.pdf")), key=lambda x: x.stat().st_size)


def main():
    print("="*70)
    print("  PDF TEXT & IMAGE EXTRACTION")
    print("="*70)
    print()
    
    pdfs = get_pdfs()
    
    if not pdfs:
        print("❌ No PDF files found")
        print(f"💡 Place PDFs in: {PDF_DIR.absolute()}")
        return
    
    print(f"📁 Found {len(pdfs)} PDF(s)\n")
    
    total_chunks = 0
    total_images = 0
    
    for pdf in pdfs:
        _, chunks, images = process_pdf(pdf)
        total_chunks += chunks
        total_images += images
    
    print()
    print("="*70)
    print(f"🎉 Extraction Complete")
    print(f"📊 Total: {total_chunks} chunks, {total_images} images from {len(pdfs)} PDFs")
    print("="*70)


if __name__ == "__main__":
    main()