# Fault Detection Video Processor

This script processes a video file using a PyTorch fault detection model, running inference on each frame and displaying the results in real-time. It supports both RGB and greyscale video input.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

Edit the configuration variables at the top of `fault_detection_video.py`:

```python
# Configuration variables - Edit these as needed
VIDEO_PATH = r"D:\Drishti\Drishti_DMRC_data\basler_1767303407.mp4"  # Path to your input video file
MODEL_PATH = r"D:\Drishti\135epochs.pt"                           # Path to your PyTorch model (.pt file)
FRAME_RATE = 35                                                  # Desired frame rate for processing
PLAYBACK_SPEED = 0.25                                            # Playback speed multiplier (0.25x = slower, 1.0x = normal, 2.0x = faster)
OUTPUT_DIR = r"D:\Drishti\Drishti_DMRC_data\Output"              # Base directory for saving detected frames
```

### Playback Speed Options:
- `0.25` = 0.25x speed (slower playback, good for detailed analysis)
- `1.0` = 1x speed (normal speed)
- `2.0` = 2x speed (faster playback)

## Features

- **Real-time fault detection** with YOLO model
- **Automatic frame saving**: Frames with detections are automatically saved to timestamped subfolders
- **Configurable playback speed** for detailed analysis
- **Live video display** with bounding boxes overlaid
- **Organized output**: Each run creates a new timestamped folder (e.g., `run_20260102_143022`)

## Controls

- Press 'q' to quit the video processing

## Customization

### Model Loading
The `load_model` function currently loads a standard PyTorch model. If your model requires special loading (e.g., with specific device or additional parameters), modify this function.

### Preprocessing
The `preprocess_frame` function assumes the model expects 224x224 input with ImageNet normalization. Adjust the `input_size` and normalization values based on your model's requirements.

### Post-processing
The `postprocess_output` function is a placeholder. Implement the actual post-processing logic based on your model's output format:

- For **classification**: Display class labels and confidence scores
- For **object detection**: Draw bounding boxes around detected faults
- For **segmentation**: Overlay segmentation masks

Example for object detection:

```python
def postprocess_output(output, frame):
    # Assuming output is [batch, num_boxes, 5] with [x1, y1, x2, y2, confidence]
    boxes = output[0]  # Remove batch dimension
    for box in boxes:
        if box[4] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = box[:4].int()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame
```

## Notes

- The script processes frames at the specified frame rate, regardless of the video's original frame rate
- Processing speed may be limited by your hardware and model complexity
- For very fast models, you can achieve real-time processing at the video's native frame rate