"""
process_all.py - Master pipeline using YOUR actual file names
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import time
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

VIDEOS_DIR = Path("videos")
PDF_DIR = Path("pdfs")
LOGS_DIR = Path("logs")
PROCESSING_LOG = LOGS_DIR / "processing_history.json"

LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# IMPORT YOUR ACTUAL MODULES
# ============================================================================

print("Loading pipeline modules...")

try:
    # Video processing (using your actual file names)
    import ingest_videos  # Your video_to_json equivalent
    import extract_frames
    import chunks_json  # Your json_to_chunks equivalent
    import chunk_embeddings
    import embed_frames
    
    # PDF processing (using your actual file names)
    import ingest_pdfs  # Your pdf_to_chunks equivalent
    import embed_pdfs  # Your embed_pdf_chunks equivalent
    
    print("✅ All modules loaded successfully\n")
    
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("\n💡 Make sure all pipeline scripts are in the same directory")
    sys.exit(1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_step(step_num: int, total_steps: int, description: str):
    """Print step progress."""
    print(f"\n{'─'*70}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'─'*70}\n")


def save_processing_log(log_data: Dict[str, Any]):
    """Save processing log to file."""
    try:
        if PROCESSING_LOG.exists():
            with open(PROCESSING_LOG, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_data)
        
        with open(PROCESSING_LOG, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"⚠️  Could not save log: {e}")


def count_files(directory: Path, extensions: tuple) -> int:
    """Count files with given extensions."""
    if not directory.exists():
        return 0
    return len([f for f in directory.iterdir() if f.suffix.lower() in extensions])


# ============================================================================
# PROCESSING STEPS
# ============================================================================

def check_input_files() -> Dict[str, int]:
    """Check what files are available."""
    print_section("📂 CHECKING INPUT FILES")
    
    video_count = count_files(VIDEOS_DIR, (".mp4", ".mkv", ".avi", ".mov", ".webm"))
    pdf_count = count_files(PDF_DIR, (".pdf",))
    
    print(f"Videos found: {video_count}")
    print(f"PDFs found: {pdf_count}")
    
    if video_count == 0 and pdf_count == 0:
        print("\n❌ No files to process!")
        print(f"💡 Add videos to: {VIDEOS_DIR.absolute()}")
        print(f"💡 Add PDFs to: {PDF_DIR.absolute()}")
        sys.exit(0)
    
    return {"videos": video_count, "pdfs": pdf_count}


def process_videos() -> Dict[str, Any]:
    """Run complete video processing pipeline."""
    print_section("🎥 VIDEO PROCESSING PIPELINE")
    
    stats = {
        "transcriptions": 0,
        "frames_extracted": 0,
        "text_chunks": 0,
        "text_embeddings": 0,
        "frame_embeddings": 0,
        "errors": []
    }
    
    try:
        # Step 1: Ingest videos
        print_step(1, 5, "Transcribing Videos (Whisper)")
        try:
            ingest_videos.main()
            json_dir = Path("json")
            if json_dir.exists():
                stats["transcriptions"] = len(list(json_dir.glob("*.json")))
        except Exception as e:
            stats["errors"].append(f"Video transcription: {e}")
        
        # Step 2: Extract frames
        print_step(2, 5, "Extracting Video Frames")
        try:
            extract_frames.main()
            frames_dir = Path("frames")
            if frames_dir.exists():
                stats["frames_extracted"] = sum(1 for _ in frames_dir.rglob("*.jpg"))
        except Exception as e:
            stats["errors"].append(f"Frame extraction: {e}")
        
        # Step 3: Chunk transcripts
        print_step(3, 5, "Chunking Transcripts")
        try:
            chunks_json.main()
            chunks_dir = Path("chunks")
            if chunks_dir.exists():
                for chunk_file in chunks_dir.glob("*_chunks.json"):
                    with open(chunk_file) as f:
                        chunks = json.load(f)
                        stats["text_chunks"] += len(chunks)
        except Exception as e:
            stats["errors"].append(f"Chunking: {e}")
        
        # Step 4: Embed text chunks
        print_step(4, 5, "Embedding Text Chunks")
        try:
            chunk_embeddings.main()
            stats["text_embeddings"] = stats["text_chunks"]
        except Exception as e:
            stats["errors"].append(f"Text embedding: {e}")
        
        # Step 5: Embed frames
        print_step(5, 5, "Embedding Video Frames")
        try:
            embed_frames.main()
            stats["frame_embeddings"] = stats["frames_extracted"]
        except Exception as e:
            stats["errors"].append(f"Frame embedding: {e}")
        
    except Exception as e:
        stats["errors"].append(f"Video pipeline critical: {e}")
    
    return stats


def process_pdfs() -> Dict[str, Any]:
    """Run complete PDF processing pipeline."""
    print_section("📄 PDF PROCESSING PIPELINE")
    
    stats = {
        "pdfs_processed": 0,
        "text_chunks": 0,
        "images_extracted": 0,
        "text_embeddings": 0,
        "image_embeddings": 0,
        "errors": []
    }
    
    try:
        # Step 1: Ingest PDFs
        print_step(1, 2, "Extracting PDF Text & Images")
        try:
            ingest_pdfs.main()
            pdf_chunks_dir = Path("pdf_chunks")
            if pdf_chunks_dir.exists():
                chunk_files = list(pdf_chunks_dir.glob("*_chunks.json"))
                stats["pdfs_processed"] = len(chunk_files)
                
                for chunk_file in chunk_files:
                    with open(chunk_file) as f:
                        chunks = json.load(f)
                        stats["text_chunks"] += len(chunks)
                
                for img_file in pdf_chunks_dir.glob("*_images.json"):
                    with open(img_file) as f:
                        images = json.load(f)
                        stats["images_extracted"] += len(images)
        except Exception as e:
            stats["errors"].append(f"PDF extraction: {e}")
        
        # Step 2: Embed PDFs
        print_step(2, 2, "Embedding PDF Content")
        try:
            embed_pdfs.main()
            stats["text_embeddings"] = stats["text_chunks"]
            stats["image_embeddings"] = stats["images_extracted"]
        except Exception as e:
            stats["errors"].append(f"PDF embedding: {e}")
    
    except Exception as e:
        stats["errors"].append(f"PDF pipeline critical: {e}")
    
    return stats


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run complete processing pipeline."""
    start_time = time.time()
    
    print("="*70)
    print("  🚀 UNIFIED PROCESSING PIPELINE")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    input_counts = check_input_files()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "input_counts": input_counts,
        "video_stats": {},
        "pdf_stats": {},
        "total_time_seconds": 0,
        "success": False
    }
    
    # Process videos
    if input_counts["videos"] > 0:
        video_stats = process_videos()
        results["video_stats"] = video_stats
    else:
        print_section("⏭️  SKIPPING VIDEO PROCESSING")
        print("No videos found\n")
    
    # Process PDFs
    if input_counts["pdfs"] > 0:
        pdf_stats = process_pdfs()
        results["pdf_stats"] = pdf_stats
    else:
        print_section("⏭️  SKIPPING PDF PROCESSING")
        print("No PDFs found\n")
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    results["total_time_seconds"] = round(total_time, 2)
    
    print_section("📊 PROCESSING COMPLETE")
    
    mins = int(total_time // 60)
    secs = int(total_time % 60)
    print(f"⏱️  Total Time: {mins}m {secs}s\n")
    
    if input_counts["videos"] > 0:
        print("🎥 Video Processing:")
        vs = results["video_stats"]
        print(f"   ✅ Transcriptions: {vs.get('transcriptions', 0)}")
        print(f"   ✅ Frames: {vs.get('frames_extracted', 0)}")
        print(f"   ✅ Text chunks: {vs.get('text_chunks', 0)}")
        print(f"   ✅ Embeddings: {vs.get('text_embeddings', 0)}")
        if vs.get('errors'):
            print(f"   ⚠️  Errors: {len(vs['errors'])}")
        print()
    
    if input_counts["pdfs"] > 0:
        print("📄 PDF Processing:")
        ps = results["pdf_stats"]
        print(f"   ✅ PDFs: {ps.get('pdfs_processed', 0)}")
        print(f"   ✅ Text chunks: {ps.get('text_chunks', 0)}")
        print(f"   ✅ Images: {ps.get('images_extracted', 0)}")
        if ps.get('errors'):
            print(f"   ⚠️  Errors: {len(ps['errors'])}")
        print()
    
    print("="*70)
    print("  🎉 READY TO QUERY!")
    print("="*70)
    print("\nNext: Run 'python unified_query.py'\n")
    
    save_processing_log(results)
    
    results["success"] = True
    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)