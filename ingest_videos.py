"""
Video to JSON Transcription with Corruption Detection
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
from datetime import datetime

# CONFIG
VIDEOS_DIR = Path("videos")
JSON_DIR = Path("json")
QUARANTINE_DIR = Path("quarantine")
WHISPER_MODEL = "base"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")
MAX_WORKERS = 1  # Parallel transcriptions (adjust based on GPU count)

JSON_DIR.mkdir(exist_ok=True)
QUARANTINE_DIR.mkdir(exist_ok=True)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_video(video_path: Path) -> Dict[str, any]:
    """
    Validate video file integrity using ffprobe.
    Returns validation result with details.
    """
    result = {
        "path": video_path,
        "valid": False,
        "error": None,
        "duration": None,
        "size_mb": None
    }
    
    # Check file exists and has size
    if not video_path.exists():
        result["error"] = "File not found"
        return result
    
    size_bytes = video_path.stat().st_size
    result["size_mb"] = round(size_bytes / (1024 * 1024), 2)
    
    if size_bytes == 0:
        result["error"] = "File is empty (0 bytes)"
        return result
    
    if size_bytes < 1024:  # Less than 1KB
        result["error"] = "File too small (likely corrupt)"
        return result
    
    # Use ffprobe to validate video structure
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path)
        ]
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if process.returncode != 0:
            result["error"] = f"ffprobe failed: {process.stderr[:100]}"
            return result
        
        probe_data = json.loads(process.stdout)
        
        # Extract duration
        if "format" in probe_data and "duration" in probe_data["format"]:
            result["duration"] = float(probe_data["format"]["duration"])
            result["valid"] = True
        else:
            result["error"] = "Missing duration info"
        
    except subprocess.TimeoutExpired:
        result["error"] = "Validation timeout (file may be corrupt)"
    except json.JSONDecodeError:
        result["error"] = "Invalid ffprobe output"
    except FileNotFoundError:
        # ffprobe not installed, do basic check only
        result["valid"] = True
        result["duration"] = "unknown"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result


def validate_json(json_path: Path) -> bool:
    """Quick JSON validation."""
    try:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        
        # Check structure
        if not isinstance(data, dict):
            return False
        
        if "segments" not in data:
            return False
        
        segments = data["segments"]
        
        if not isinstance(segments, list) or len(segments) == 0:
            return False
        
        # Check first and last segment for required fields
        for seg in [segments[0], segments[-1]]:
            if not all(k in seg for k in ["start", "end", "text"]):
                return False
        
        return True
        
    except:
        return False


def quarantine_file(file_path: Path, reason: str):
    """Move corrupt file to quarantine with metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine_subdir = QUARANTINE_DIR / timestamp
    quarantine_subdir.mkdir(exist_ok=True)
    
    # Move file
    dest = quarantine_subdir / file_path.name
    shutil.move(str(file_path), str(dest))
    
    # Create metadata
    metadata = {
        "original_path": str(file_path),
        "quarantined_at": timestamp,
        "reason": reason,
        "size_bytes": dest.stat().st_size
    }
    
    metadata_file = quarantine_subdir / f"{file_path.stem}_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   🗑️  Quarantined: {file_path.name}")


def validate_all_videos(videos: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    Validate all video files.
    Returns (valid_videos, corrupt_videos).
    """
    print("🔍 Pre-flight check: Validating videos...")
    print()
    
    valid = []
    corrupt = []
    
    for video in videos:
        print(f"   Checking: {video.name}... ", end="", flush=True)
        result = validate_video(video)
        
        if result["valid"]:
            duration_str = f"{result['duration']:.1f}s" if isinstance(result['duration'], float) else result['duration']
            print(f"✅ Valid ({duration_str}, {result['size_mb']}MB)")
            valid.append(video)
        else:
            print(f"❌ {result['error']}")
            corrupt.append(video)
    
    return valid, corrupt


# ============================================================================
# TRANSCRIPTION FUNCTIONS
# ============================================================================

def transcribe_video(video_path: Path) -> tuple[Path, bool]:
    """Transcribe video and return (path, success)."""
    output_json = JSON_DIR / f"{video_path.stem}.json"
    
    # Validate existing JSON
    if output_json.exists():
        if validate_json(output_json):
            print(f"⏭️  Skip: {video_path.name}")
            return (video_path, True)
        else:
            print(f"⚠️  Re-transcribing (corrupt JSON): {video_path.name}")
            output_json.unlink()
    
    print(f"🎙️  Start: {video_path.name}")
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            subprocess.run([
                "python", "-m", "whisper",
                str(video_path),
                "--model", WHISPER_MODEL,
                "--output_format", "json",
                "--output_dir", str(JSON_DIR),
                "--language", "en",
                "--fp16", "False"
            ], check=True, capture_output=True, timeout=3600)
            
            # Validate output
            if output_json.exists() and validate_json(output_json):
                print(f"✅ Done: {video_path.name}")
                return (video_path, True)
            else:
                print(f"⚠️  Output validation failed (attempt {attempt + 1}/{max_retries})")
                if output_json.exists():
                    output_json.unlink()
                
        except subprocess.TimeoutExpired:
            print(f"⚠️  Timeout (attempt {attempt + 1}/{max_retries})")
            if output_json.exists():
                output_json.unlink()
        except Exception as e:
            print(f"⚠️  Error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}")
            if output_json.exists():
                output_json.unlink()
    
    print(f"❌ Failed after {max_retries} attempts: {video_path.name}")
    return (video_path, False)


def get_videos() -> List[Path]:
    """Get all video files sorted by size (smallest first)."""
    if not VIDEOS_DIR.exists():
        return []
    videos = [f for f in VIDEOS_DIR.iterdir() 
              if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos, key=lambda x: x.stat().st_size)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("  VIDEO TO JSON TRANSCRIPTION")
    print("="*70)
    print()
    
    # Get all videos
    all_videos = get_videos()
    
    if not all_videos:
        print("⚠️  No videos found")
        return
    
    # Validate videos
    valid_videos, corrupt_videos = validate_all_videos(all_videos)
    
    print()
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"✅ Valid videos: {len(valid_videos)}")
    print(f"❌ Corrupt videos: {len(corrupt_videos)}")
    print()
    
    # Handle corrupt videos
    if corrupt_videos:
        print("🗑️  Corrupt videos found:")
        for video in corrupt_videos:
            print(f"   - {video.name}")
        print()
        
        choice = input("Move corrupt videos to quarantine? (y/n): ").strip().lower()
        if choice == 'y':
            for video in corrupt_videos:
                result = validate_video(video)
                quarantine_file(video, result["error"])
            print()
    
    # Process valid videos
    if not valid_videos:
        print("⚠️  No valid videos to process")
        return
    
    print("="*70)
    print("STARTING TRANSCRIPTION")
    print("="*70)
    print()
    
    success = 0
    failed = 0
    
    # Parallel processing
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(transcribe_video, video): video for video in valid_videos}
        
        for future in as_completed(futures):
            _, succeeded = future.result()
            if succeeded:
                success += 1
            else:
                failed += 1
    
    print()
    print("="*70)
    print("TRANSCRIPTION COMPLETE")
    print("="*70)
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {failed}")
    if corrupt_videos:
        print(f"🗑️  Quarantined: {len(corrupt_videos)}")
    print("="*70)


if __name__ == "__main__":
    main()