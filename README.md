# AI-Powered Material Loading Counter

This project uses computer vision with YOLOv8 to automatically detect a lorry in a video and count materials (like metal sheets or steel spans) as they are loaded onto it. It generates an annotated output video showing the detections and the live count.

## Features

- **Automatic Lorry Detection**: The script first analyzes the video to find a stable bounding box for the lorry, which defines the loading zone.
- **Object Detection**: Uses custom-trained YOLOv8 models to detect specific materials in each frame.
- **Intelligent Object Merging**: Fixes a common issue where large objects are detected as multiple smaller pieces. It merges overlapping bounding boxes into a single entity before tracking.
- **Object Tracking & Counting**: Tracks each detected item and increments a counter only when the item's center point enters the lorry zone for the first time.
- **Annotated Video Output**: Creates a new video file with visual overlays for the lorry zone, detected items, and the running count.

## Requirements

- Python 3.8+
- The Python libraries listed in `requirements.txt`

### Pre-trained Model Files

You will also need the following pre-trained model files:
- `yolov8l.pt` (for general lorry detection)
- `best.pt` or `best1.pt` (for counting metal sheets)
- `best_span.pt` (for counting steel spans)


3. **Place the models**: Make sure your pre-trained YOLO model files (`best.pt`, `best1.pt`, `best_span.pt`, and `yolov8l.pt`) are accessible from the script, either by placing them in the same directory or providing their full path.

## How to Run

To run the script for different materials, you need to modify two variables in the `main()` function of the Python script: `metal_model` and `output_path`.

Follow the instructions for the specific material you want to count.

### Scenario 1: Counting Metal Sheets (Run for Count 2)

For this scenario, we will use the model trained for general metal sheets and save the output as `count2_merged.mp4`.

**Steps:**
1. Open the Python script (e.g., `count2.py`)
3. Modify the `metal_model` path to point to your metal sheet model (`best.pt` or `best1.pt`)
4. Modify the `output_path` to save the video as `count2_merged.mp4`

**Key modifications needed:**
- Set output path for Run Count 2
- Set model path for Metal Sheets (use `best.pt` or `best1.pt` depending on your best model)

### Scenario 2: Counting Steel Spans (Run for Count 3)

For this scenario, we will use the model specifically trained for steel spans and save the output as `count3_merged.mp4`.

**Steps:**
1. Open the Python script(`count3.py`)
2. Navigate to the `main()` function
3. Modify the `metal_model` path to point to your steel span model (`best_span.pt`)
4. Modify the `output_path` to save the video as `count3_merged.mp4`

**Key modifications needed:**
- Set output path for Run Count 3
- Set model path for Steel Spans

## Running the Script

After making the changes for your desired scenario, simply run the script from your terminal.

A window will pop up showing the video processing in real-time. Press 'q' to stop the process early. The final annotated video will be saved to the specified output path.

## Output

The system will generate an annotated video file showing:
- The detected lorry zone
- Real-time material detection and counting
- Visual overlays for tracked objects
- Running count display