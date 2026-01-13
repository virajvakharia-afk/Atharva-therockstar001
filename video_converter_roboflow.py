import cv2
import os
from datetime import datetime
import subprocess

def convert_video_for_roboflow(input_path, output_dir=r"D:\Drishti\Drishti_DMRC_data\converted_videos"):
    """
    Convert a video file to MP4 format with H.264 codec for Roboflow compatibility.

    Args:
        input_path (str): Path to the input video file
        output_dir (str): Directory to save the converted video (default: "converted_videos")

    Returns:
        str: Path to the converted video file, or None if conversion failed
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get filename without extension
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)

    # Create output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{name}_roboflow_{timestamp}.mp4")

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Ensure reasonable frame rate (Roboflow prefers 30 FPS or less)
    if fps > 30:
        fps = 30.0
        print(f"Adjusting FPS from {cap.get(cv2.CAP_PROP_FPS)} to 30 for Roboflow compatibility")

    print(f"Converting video: {width}x{height} at {fps} FPS")
    print(f"Output will be saved to: {output_path}")

    # Release cap as we don't need it for FFmpeg
    cap.release()

    # FFmpeg command for conversion
    ffmpeg_path = r"C:\Users\Viraj\Downloads\ffmpeg-8.0-full_build\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
    ffmpeg_cmd = [ffmpeg_path, '-i', input_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-r', str(int(fps)), output_path]

    print(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Conversion complete!")
        print(f"Converted video saved to: {output_path}")
        return output_path
    else:
        print("Error: FFmpeg conversion failed")
        print("stderr:", result.stderr)
        return None

if __name__ == "__main__":
    # Example usage - change this path to your video file
    input_video_path = r"D:\Drishti\Drishti_DMRC_data\basler_1767303407.mp4"

    # Convert the video
    result = convert_video_for_roboflow(input_video_path)

    if result:
        print(f"\nSuccess! Your Roboflow-compatible video is ready at: {result}")
        print("You can now upload this video to Roboflow for annotation or training.")
    else:
        print("\nConversion failed. Please check the input video path and try again.")