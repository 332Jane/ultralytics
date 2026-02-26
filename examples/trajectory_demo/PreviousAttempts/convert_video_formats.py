"""
[DEPRECATED] convert_video_formats.py

Convert AVI video to multiple formats
"""

import cv2
import sys
from pathlib import Path

def convert_video(input_path, output_path):
    """Convert video format"""
    
    print(f"Reading: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"❌ Unable to open input video")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input info: {width}×{height}, {total_frames}frames, {fps:.2f}FPS")
    
    # Get extension and select codec
    ext = Path(output_path).suffix.lower()
    
    if ext == '.mp4':
        # MP4 - Try multiple codecs
        codecs = [
            cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.VideoWriter_fourcc(*'avc1'),
            cv2.VideoWriter_fourcc(*'H264'),
        ]
    elif ext == '.avi':
        # AVI
        codecs = [
            cv2.VideoWriter_fourcc(*'MJPG'),  # Motion JPEG
            cv2.VideoWriter_fourcc(*'DIVX'),
            cv2.VideoWriter_fourcc(*'XVID'),
        ]
    elif ext == '.mov':
        # MOV
        codecs = [
            cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.VideoWriter_fourcc(*'avc1'),
        ]
    else:
        print(f"❌ Unsupported format: {ext}")
        return False
    
    # Try codecs
    out = None
    for codec in codecs:
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))
        if out.isOpened():
            codec_name = chr(codec & 0xff) + chr((codec >> 8) & 0xff) + chr((codec >> 16) & 0xff) + chr((codec >> 24) & 0xff)
            print(f"✓ Using codec: {codec_name.strip()}")
            break
    
    if out is None or not out.isOpened():
        print(f"❌ Unable to create output video")
        cap.release()
        return False
    
    # Copy frame by frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Progress: {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    print(f"✓ Conversion completed: {output_path}")
    print(f"  Processed {frame_count} frames")
    
    return True


if __name__ == '__main__':
    input_file = '/workspace/ultralytics/videos/warped_test_360x2400.avi'
    
    # Convert to multiple formats
    outputs = [
        '/workspace/ultralytics/videos/warped_test_360x2400.mov',  # MOV format
        '/workspace/ultralytics/videos/warped_test_360x2400_mjpg.avi',  # MJPG encoded AVI
    ]
    
    print("="*70)
    print("Video Format Conversion")
    print("="*70)
    
    for output in outputs:
        print(f"\nConverting to: {output}")
        convert_video(input_file, output)
    
    print("\n" + "="*70)
    print("✓ Conversion completed")
    print("="*70)
