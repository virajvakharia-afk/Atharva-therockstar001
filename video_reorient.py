import cv2
import os
from datetime import datetime

# ============================================================================
# Configuration variables - Edit these as needed
# ============================================================================

# Input: Can be a single video file OR a directory containing multiple videos
INPUT_PATH = r"D:\Drishti\Drishti_DMRC_data\feed1.avi"

# Rotation direction: "clockwise" (90° right) or "counterclockwise" (90° left)
ROTATION_DIRECTION = "clockwise"

# Output directory (will be created if it doesn't exist)
OUTPUT_DIR = r"D:\Drishti\Drishti_DMRC_data\Re-oriented"

# Video file extensions to process (when INPUT_PATH is a directory)
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm')

# ============================================================================


def get_rotation_code(direction):
    """
    Get OpenCV rotation code based on direction string.
    """
    if direction.lower() == "clockwise":
        return cv2.ROTATE_90_CLOCKWISE
    elif direction.lower() == "counterclockwise":
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        raise ValueError(f"Invalid rotation direction: {direction}. Use 'clockwise' or 'counterclockwise'")


def get_video_info(video_path):
    """
    Get video properties: FPS, width, height, total frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    return fps, width, height, total_frames


def reorient_video(input_path, output_path, rotation_direction):
    """
    Rotate a video by 90 degrees and save to output path.
    Preserves original FPS and quality.
    """
    # Get original video properties
    fps, width, height, total_frames = get_video_info(input_path)
    
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"{'='*60}")
    print(f"  Input resolution: {width}x{height}")
    print(f"  Input FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Rotation: {rotation_direction}")
    
    # After 90° rotation, width and height are swapped
    new_width = height
    new_height = width
    print(f"  Output resolution: {new_width}x{new_height}")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open video file")
        return False
    
    # Create video writer with same FPS as input
    # Using mp4v codec for good quality and compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    if not out.isOpened():
        print(f"  ERROR: Could not create output video writer")
        cap.release()
        return False
    
    # Get rotation code
    rotation_code = get_rotation_code(rotation_direction)
    
    frame_count = 0
    print(f"  Processing frames...", end='', flush=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rotate the frame
        rotated_frame = cv2.rotate(frame, rotation_code)
        
        # Write to output
        out.write(rotated_frame)
        frame_count += 1
        
        # Progress indicator every 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"\r  Processing frames... {frame_count}/{total_frames} ({progress:.1f}%)", end='', flush=True)
    
    print(f"\r  Processing frames... {frame_count}/{total_frames} (100.0%)")
    
    # Release resources
    cap.release()
    out.release()
    
    # Verify output file
    if os.path.exists(output_path):
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"  Output saved: {output_path}")
        print(f"  Output size: {output_size:.2f} MB")
        return True
    else:
        print(f"  ERROR: Output file was not created")
        return False


def get_video_files(path):
    """
    Get list of video files from path (single file or directory).
    """
    if os.path.isfile(path):
        if path.lower().endswith(VIDEO_EXTENSIONS):
            return [path]
        else:
            print(f"Warning: {path} is not a recognized video file")
            return []
    elif os.path.isdir(path):
        video_files = []
        for filename in os.listdir(path):
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                video_files.append(os.path.join(path, filename))
        return sorted(video_files)
    else:
        print(f"Error: {path} does not exist")
        return []


def main():
    print("\n" + "="*60)
    print("VIDEO REORIENTATION TOOL")
    print("="*60)
    print(f"Input path: {INPUT_PATH}")
    print(f"Rotation: {ROTATION_DIRECTION}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of videos to process
    video_files = get_video_files(INPUT_PATH)
    
    if not video_files:
        print("\nNo video files found to process!")
        return
    
    print(f"\nFound {len(video_files)} video(s) to process")
    
    # Process each video
    successful = 0
    failed = 0
    
    for video_path in video_files:
        # Generate output filename
        base_name = os.path.basename(video_path)
        name, ext = os.path.splitext(base_name)
        output_filename = f"{name}_reoriented.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Reorient the video
        if reorient_video(video_path, output_path, ROTATION_DIRECTION):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Total videos: {len(video_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

