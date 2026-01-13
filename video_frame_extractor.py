import cv2
import os
from datetime import datetime

# Configuration variables - Edit these as needed
VIDEO_PATH = r"D:\Drishti\Drishti_DMRC_data\Re-oriented\feed1_reoriented.mp4"  # Path to your input video file
OUTPUT_DIR = r"D:\Drishti\Drishti_DMRC_data\Extracted_Frames"                   # Base directory for saving extracted frames
IMAGE_FORMAT = "jpg"                                                            # Output image format: jpg, png, bmp
JPEG_QUALITY = 95                                                               # JPEG quality (1-100), only used if IMAGE_FORMAT is jpg


def extract_frames(video_path, output_dir, image_format="jpg", jpeg_quality=95):
    """
    Extract all frames from a video file and save them to the output directory.
    
    Args:
        video_path: Path to the input video file
        output_dir: Base directory for saving extracted frames
        image_format: Output image format (jpg, png, bmp)
        jpeg_quality: JPEG quality (1-100), only used for jpg format
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print("Warning: Could not detect video FPS, defaulting to 30")
        video_fps = 30
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output directory with video name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = os.path.join(output_dir, f"{video_name}_{timestamp}")
    os.makedirs(frames_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("VIDEO FRAME EXTRACTOR")
    print("=" * 60)
    print(f"\nVideo Info:")
    print(f"  File: {os.path.basename(video_path)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"\nOutput:")
    print(f"  Directory: {frames_dir}")
    print(f"  Format: {image_format.upper()}")
    if image_format.lower() == "jpg":
        print(f"  JPEG Quality: {jpeg_quality}")
    print("\n" + "-" * 60)
    print("Extracting frames...")
    print("-" * 60)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Generate filename with zero-padded frame number
        # Calculate padding based on total frames
        padding = len(str(total_frames))
        filename = f"frame_{frame_count:0{padding}d}.{image_format}"
        filepath = os.path.join(frames_dir, filename)
        
        # Save the frame
        if image_format.lower() == "jpg":
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        elif image_format.lower() == "png":
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(filepath, frame)
        
        saved_count += 1
        
        # Print progress every 100 frames or at specific percentages
        if frame_count % 100 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"  Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Clean up
    cap.release()
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Total frames extracted: {saved_count}")
    print(f"  Output directory: {frames_dir}")
    print(f"  Frames per second in source: {video_fps:.2f}")
    print("=" * 60 + "\n")
    
    return frames_dir


def main():
    """Main function to run the frame extraction."""
    extract_frames(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        image_format=IMAGE_FORMAT,
        jpeg_quality=JPEG_QUALITY
    )


if __name__ == "__main__":
    main()

