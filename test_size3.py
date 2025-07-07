import cv2
import numpy as np
from ultralytics import YOLO
import os
import math

# --- Constants ---
# Set the known height of the reference person in feet.
REFERENCE_PERSON_HEIGHT_FEET = 5.3
# Confidence threshold for a keypoint to be considered "visible".
KEYPOINT_CONF_THRESHOLD = 0.5 

# --- Model Paths ---
# Ensure these paths are correct for your system.
POSE_MODEL_PATH = "yolov8n-pose.pt" 
STEEL_SEG_MODEL_PATH = r"D:\steel\runs\segment\steel_detector_seg3\weights\best.pt" 

def get_pixel_height_from_pose(keypoints):
    """
    Calculates the vertical distance in pixels from the nose to the ankles.

    This method is specifically chosen to be robust against posture changes like
    raised arms. By measuring from the nose (a proxy for the head) to the
    ankles, we get a consistent measure of body height that is not affected by
    hand position.

    Args:
        keypoints (torch.Tensor): A tensor of keypoints for a single person.
                                  Shape: (17, 3) where each row is [x, y, confidence].
    Returns:
        float or None: The height in pixels, or None if keypoints are not visible.
    """
    # Keypoint indices for COCO format: 0=nose, 15=left_ankle, 16=right_ankle
    nose_kpt = keypoints[0]
    left_ankle_kpt = keypoints[15]
    right_ankle_kpt = keypoints[16]

    # We use the nose (keypoint 0) as the top reference point. This is a deliberate
    # choice because it's consistently detected and represents the head area.
    # Crucially, it IGNORES the position of the hands, so if a person raises
    # their arms, it will not incorrectly increase their measured height.
    if nose_kpt[2] < KEYPOINT_CONF_THRESHOLD:
        return None # Cannot see the person's head clearly.

    # Check ankle visibility and use the available one(s)
    left_ankle_visible = left_ankle_kpt[2] > KEYPOINT_CONF_THRESHOLD
    right_ankle_visible = right_ankle_kpt[2] > KEYPOINT_CONF_THRESHOLD

    avg_ankle_y = 0
    if left_ankle_visible and right_ankle_visible:
        # Best case: both ankles are visible, average them for stability.
        avg_ankle_y = (left_ankle_kpt[1] + right_ankle_kpt[1]) / 2
    elif left_ankle_visible:
        # Fallback: use only the left ankle.
        avg_ankle_y = left_ankle_kpt[1]
    elif right_ankle_visible:
        # Fallback: use only the right ankle.
        avg_ankle_y = right_ankle_kpt[1]
    else:
        # Cannot determine height without a stable foot position.
        return None

    # Calculate height as the absolute difference in y-coordinates.
    pixel_height = abs(nose_kpt[1] - avg_ankle_y)
    return pixel_height

def find_closest_person_scale(steel_center, pose_results):
    """
    Finds the person closest to the steel object and calculates their pixels_per_foot scale.
    
    This ensures that the reference object (the person) is at a similar depth
    in the scene as the object being measured, improving accuracy.

    Args:
        steel_center (tuple): (x, y) coordinates of the steel's center.
        pose_results (list): List of YOLO results containing person poses.

    Returns:
        float or None: The calculated pixels_per_foot scale.
        tuple or None: The bounding box (xyxy) of the reference person.
    """
    min_dist = float('inf')
    best_scale = None
    ref_person_box = None
    
    # Check if there are any detected poses
    if not pose_results or pose_results[0].keypoints.shape[0] == 0:
        return None, None

    # Iterate through each detected person in the frame
    for i, person_kpts in enumerate(pose_results[0].keypoints.data):
        person_pixel_height = get_pixel_height_from_pose(person_kpts)
        
        if person_pixel_height:
            # Calculate the person's center based on nose and ankles for distance check
            nose_pt = person_kpts[0][:2]
            avg_ankle_y = (person_kpts[15][1] + person_kpts[16][1]) / 2
            person_center_x = nose_pt[0]
            person_center_y = (nose_pt[1] + avg_ankle_y) / 2
            
            # Calculate Euclidean distance from steel center to person center
            dist = math.hypot(steel_center[0] - person_center_x, steel_center[1] - person_center_y)

            if dist < min_dist:
                min_dist = dist
                best_scale = person_pixel_height / REFERENCE_PERSON_HEIGHT_FEET
                # The box and keypoints have the same order, so we can use the index 'i'
                ref_person_box = pose_results[0].boxes.xyxy[i]

    return best_scale, ref_person_box


def main():
    video_path = r"C:\Users\STUDENT\Downloads\RR constructions centring material-20250704T045212Z-1-003\RR constructions centring material\Centring sheet\Copy of VID_20250605_111315.mp4"
    output_path = r"D:\steel\output\steel_measurement_output_v2.mp4"

    print("[INFO] Loading YOLO models...")
    try:
        pose_model = YOLO(POSE_MODEL_PATH)
        steel_model = YOLO(STEEL_SEG_MODEL_PATH)
        print("[INFO] Models loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load models - {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"[INFO] Video processing started. Output will be saved to {output_path}")

    frame_num = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] End of video reached.")
                break
            
            frame_num += 1
            # Process every frame, but print status less frequently
            if frame_num % 30 == 0:
                print(f"[INFO] Processing frame {frame_num}...")

            # Run pose estimation to find all people in the frame
            pose_results = pose_model(frame, classes=0, verbose=False) # class 0 is 'person'

            # Run segmentation to find all steel objects
            steel_results = steel_model(frame, conf=0.4, verbose=False)

            annotated_frame = frame.copy()
            
            # Only proceed if steel objects are detected
            if steel_results[0].masks:
                polygons = steel_results[0].masks.xy
                for poly in polygons:
                    contour = np.array(poly, dtype=np.int32)
                    
                    # Get the minimum area rectangle for the steel object
                    rect = cv2.minAreaRect(contour)
                    (w_px, h_px) = rect[1]
                    center_px = tuple(np.array(rect[0]).astype(int))
                    box_points = np.array(cv2.boxPoints(rect)).astype(int)

                    cv2.drawContours(annotated_frame, [box_points], 0, (0, 255, 0), 2)
                    
                    # Find the closest person to use as a scale reference
                    pixels_per_foot, ref_person_box = find_closest_person_scale(center_px, pose_results)
                    
                    dim_text = "No reference person" 
                    
                    if pixels_per_foot and pixels_per_foot > 0:
                        w_ft = w_px / pixels_per_foot
                        h_ft = h_px / pixels_per_foot
                        
                        # Consistently label the longer side as "Width" and shorter as "Height"
                        width_ft = max(w_ft, h_ft)
                        height_ft = min(w_ft, h_ft)

                        dim_text = f"W: {width_ft:.2f} ft, H: {height_ft:.2f} ft"
                        
                        # Visualize the reference person and the link to the object
                        if ref_person_box is not None:
                            p1 = (int(ref_person_box[0]), int(ref_person_box[1]))
                            p2 = (int(ref_person_box[2]), int(ref_person_box[3]))
                            person_center = ( (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 )
                            
                            # Draw a line connecting the steel and the reference person
                            cv2.line(annotated_frame, center_px, person_center, (255, 255, 0), 2)
                            # Draw a box around the reference person
                            cv2.rectangle(annotated_frame, p1, p2, (255, 255, 0), 2)
                            cv2.putText(annotated_frame, "Reference", (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # Display the dimension text on the steel object
                    cv2.putText(annotated_frame, dim_text, (center_px[0] - 80, center_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(annotated_frame)
            # Display the output in a reasonably sized window
            cv2.imshow('Steel Measurement', cv2.resize(annotated_frame, (1280, 720)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User terminated the process.")
                break

    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Processing finished. Video saved to {output_path}")

if __name__ == "__main__":
    main()