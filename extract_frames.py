"""
extract_frames.py - Extract keyframes from videos
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib

# CONFIG
VIDEOS_DIR = Path("videos")
FRAMES_DIR = Path("frames")
JSON_DIR = Path("json")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")

# Frame extraction settings
FRAME_INTERVAL = 30  # Extract 1 frame every 30 seconds
MAX_FRAMES_PER_VIDEO = 200  # Limit frames to prevent storage issues
FRAME_QUALITY = 2  # JPEG quality (1=best, 31=worst)

FRAMES_DIR.mkdir(exist_ok=True)


def extract_frames(video_path: Path, transcript_path: Path) -> Tuple[str, int]:
    """
    Extract keyframes from video at regular intervals.
    Returns (video_name, num_frames).
    """
    video_name = video_path.stem
    frame_subdir = FRAMES_DIR / video_name
    frame_subdir.mkdir(exist_ok=True)
    
    # Check if already processed
    metadata_file = frame_subdir / f"{video_name}_frames_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                existing = json.load(f)
            print(f"⏭️  Skip: {video_name} ({len(existing)} frames)")
            return (video_name, len(existing))
        except:
            print(f"⚠️  Re-extracting (corrupt metadata): {video_name}")
    
    # Load transcript to get duration
    try:
        with open(transcript_path, encoding='utf-8') as f:
            transcript = json.load(f)
    except:
        print(f"❌ Cannot read transcript: {video_name}")
        return (video_name, 0)
    
    segments = transcript.get("segments", [])
    if not segments:
        print(f"⚠️  No segments in transcript: {video_name}")
        return (video_name, 0)
    
    duration = segments[-1]["end"]
    
    # Calculate frame timestamps
    num_frames = min(int(duration / FRAME_INTERVAL), MAX_FRAMES_PER_VIDEO)
    
    print(f"🎬 Extracting {num_frames} frames from {video_name} ({duration:.1f}s)")
    
    frames_metadata = []
    extracted = 0
    
    for i in range(num_frames):
        timestamp = i * FRAME_INTERVAL
        
        # Generate frame filename
        frame_hash = hashlib.md5(f"{video_name}_{timestamp}".encode()).hexdigest()[:8]
        frame_filename = f"{video_name}_frame_{i:04d}_{frame_hash}.jpg"
        frame_path = frame_subdir / frame_filename
        
        # Skip if already exists
        if frame_path.exists():
            frames_metadata.append({
                "video": video_name,
                "frame_index": i,
                "timestamp": timestamp,
                "path": str(frame_path)
            })
            continue
        
        # Extract frame using ffmpeg
        try:
            cmd = [
                "ffmpeg",
                "-ss", str(timestamp),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", str(FRAME_QUALITY),
                "-y",
                str(frame_path)
            ]
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=30
            )
            
            frames_metadata.append({
                "video": video_name,
                "frame_index": i,
                "timestamp": timestamp,
                "path": str(frame_path)
            })
            extracted += 1
            
        except subprocess.TimeoutExpired:
            print(f"  ⚠️  Timeout extracting frame at {timestamp}s")
        except FileNotFoundError:
            print(f"  ❌ ffmpeg not found. Install: sudo apt install ffmpeg")
            return (video_name, 0)
        except Exception as e:
            print(f"  ⚠️  Failed at {timestamp}s: {type(e).__name__}")
    
    if extracted > 0:
        print(f"  ✅ Extracted {extracted} new frames (total: {len(frames_metadata)})")
    else:
        print(f"  ✅ All {len(frames_metadata)} frames already exist")
    
    # Save metadata
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(frames_metadata, f, indent=2)
    
    return (video_name, len(frames_metadata))


def get_videos_with_transcripts() -> List[Tuple[Path, Path]]:
    """Get videos that have corresponding transcripts."""
    if not VIDEOS_DIR.exists() or not JSON_DIR.exists():
        return []
    
    pairs = []
    for json_file in JSON_DIR.glob("*.json"):
        video_name = json_file.stem
        
        # Find corresponding video
        for ext in VIDEO_EXTENSIONS:
            video_path = VIDEOS_DIR / f"{video_name}{ext}"
            if video_path.exists():
                pairs.append((video_path, json_file))
                break
    
    return sorted(pairs, key=lambda x: x[0].stat().st_size)


def main():
    print("="*70)
    print("  VIDEO FRAME EXTRACTION")
    print("="*70)
    print()
    
    video_transcript_pairs = get_videos_with_transcripts()
    
    if not video_transcript_pairs:
        print("❌ No videos with transcripts found")
        print("💡 Run video_to_json.py first")
        return
    
    print(f"📁 Found {len(video_transcript_pairs)} video(s) with transcripts\n")
    
    total_frames = 0
    
    for video_path, transcript_path in video_transcript_pairs:
        _, num_frames = extract_frames(video_path, transcript_path)
        total_frames += num_frames
    
    print()
    print("="*70)
    print(f"🎉 Extraction Complete: {total_frames} total frames")
    print("="*70)


if __name__ == "__main__":
    main()