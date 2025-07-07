import cv2
import numpy as np
from ultralytics import YOLO
import os

def calculate_and_draw_dimensions(frame, results):
    """
    Processes segmentation results to find, measure, and draw steel sheets.

    Args:
        frame (np.ndarray): The current video frame to draw on.
        results (ultralytics.engine.results.Results): The results from the YOLO model.
    """
    # The model's results object will contain masks if it's a segmentation model
    if results.masks is None:
        return frame, 0 # Return early if no masks are found

    # Get the polygon points for each mask
    # results.masks.xy is a list of polygons, where each polygon is a numpy array of points
    polygons = results.masks.xy
    
    # Keep track of number of detected sheets in this frame
    detected_count = len(polygons)

    for i, poly in enumerate(polygons):
        # The polygon points need to be integers for cv2 functions
        contour = np.array(poly, dtype=np.int32)

        # 1. Calculate the minimum area rotated rectangle for the contour
        # This gives a tight-fitting box even if the object is tilted
        # rect is a tuple: ((center_x, center_y), (width, height), angle)
        rect = cv2.minAreaRect(contour)
        
        # 2. Get the dimensions (width and height) from the rectangle
        # Note: cv2.minAreaRect doesn't guarantee which dimension is 'width' or 'height',
        # it's just the dimensions of the rectangle's sides.
        (width, height) = rect[1]

        # 3. Get the 4 corner points of the rotated rectangle for drawing
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points) # Convert points to integers

        # 4. Draw the rotated rectangle (contour) on the frame
        # We draw it in green with a thickness of 2
        cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 2)

        # 5. Prepare and display the dimension text
        # We display width and height with one decimal place
        dim_text = f"W: {width:.1f}, H: {height:.1f}"
        
        # Position the text near the center of the rectangle
        center_point = (int(rect[0][0]), int(rect[0][1]))
        
        # Put a white background behind the text for better readability
        text_size, _ = cv2.getTextSize(dim_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_bg_tl = (center_point[0] - text_size[0] // 2 - 5, center_point[1] - text_size[1] // 2 - 5)
        text_bg_br = (center_point[0] + text_size[0] // 2 + 5, center_point[1] + text_size[1] // 2 + 5)
        cv2.rectangle(frame, text_bg_tl, text_bg_br, (255, 255, 255), -1)

        # Write the text on the frame in blue
        cv2.putText(
            frame,
            dim_text,
            (center_point[0] - text_size[0] // 2, center_point[1] + text_size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0), # Blue color for text
            2
        )
        
    return frame, detected_count

def main():
    # --- Configuration ---
    # IMPORTANT: Update these paths to match your system
    video_path = r"C:\Users\STUDENT\Downloads\Copy of VID_20250605_123533.mp4"
    output_path = r"D:\steel\output\deep_track_lorry_count.mp4"
    model_path = r"D:\steel\runs\segment\steel_detector_seg3\weights\best.pt" 

    print("=" * 60)
    print("STEEL SHEET DIMENSION MEASUREMENT")
    print(f"Using segmentation model: {model_path}")
    print("=" * 60)

    # Load your trained YOLO segmentation model
    print("[INFO] Loading YOLO segmentation model...")
    try:
        model = YOLO(model_path)
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load model - {e}")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file at {video_path}")
        return

    # Get video properties for the output writer
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- THIS IS THE CORRECTED SECTION ---
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # ------------------------------------
    
    print(f"[INFO] Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Setup video writer
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"ERROR: Cannot create output video file at {output_path}")
        cap.release()
        return

    frame_num = 0
    print("\n[INFO] Starting video processing to measure steel dimensions...")
    print("Press 'q' in the display window to stop early.")
    print("="*60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] Reached end of video.")
                break
            
            frame_num += 1

            # Show progress in the console
            if frame_num % (fps * 2) == 0: # Print update every 2 seconds
                progress = (frame_num / total_frames) * 100
                print(f"[INFO] Progress: {progress:.1f}% (Frame: {frame_num}/{total_frames})")

            # Perform inference with the segmentation model
            results = model(frame, imgsz=640, conf=0.4, verbose=False)[0]

            # Create a copy of the frame to draw on
            annotated_frame = frame.copy()
            
            # Process results to find dimensions and draw on the frame
            annotated_frame, detected_count = calculate_and_draw_dimensions(annotated_frame, results)

            # --- Display information on the frame ---
            info_text = f"Detected Sheets: {detected_count}"
            cv2.putText(annotated_frame, info_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            progress_text = f"Frame: {frame_num}/{total_frames} ({ (frame_num/total_frames)*100:.1f}%)"
            cv2.putText(annotated_frame, progress_text, (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Write the annotated frame to the output video file
            out.write(annotated_frame)

            # Display the resulting frame in a window
            display_height = 720
            display_width = int(width * (display_height / height))
            display_frame = cv2.resize(annotated_frame, (display_width, display_height))
            cv2.imshow('Steel Sheet Dimension Measurement - Press Q to exit', display_frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] User requested exit.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user.")
    
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Release all resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total frames processed: {frame_num}")
        print(f"Output video saved to: {output_path}")
        print("="*60)
if __name__ == "__main__":
    main()
