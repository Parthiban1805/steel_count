import cv2
import numpy as np
from ultralytics import YOLO
import os
import math

REFERENCE_PERSON_HEIGHT_FEET = 5.3
POSE_MODEL_PATH = "yolov8n-pose.pt" 
STEEL_SEG_MODEL_PATH = r"D:\steel\runs\segment\steel_detector_seg3\weights\best.pt" 

def get_pixel_height_from_pose(keypoints):
    """
    Calculates the vertical distance in pixels from the nose to the ankles.
    This is a more stable measure of height than a bounding box.
    Args:
        keypoints (torch.Tensor): A tensor of keypoints for a single person.
                                  Shape: (17, 3) where each row is [x, y, confidence].
    Returns:
        float or None: The height in pixels, or None if keypoints are not visible.
    """
    nose_kpt = keypoints[0]
    left_ankle_kpt = keypoints[15]
    right_ankle_kpt = keypoints[16]

    if nose_kpt[2] > 0.5 and left_ankle_kpt[2] > 0.5 and right_ankle_kpt[2] > 0.5:
        # Average the ankle y-coordinates for a more stable base point
        avg_ankle_y = (left_ankle_kpt[1] + right_ankle_kpt[1]) / 2
        # Height is the difference in y-coordinates
        pixel_height = abs(nose_kpt[1] - avg_ankle_y)
        return pixel_height
    return None

def find_closest_person_scale(steel_center, pose_results):
    """
    Finds the person closest to the steel and calculates their pixels_per_foot scale.
    Args:
        steel_center (tuple): (x, y) coordinates of the steel's center.
        pose_results (list): List of YOLO results for people poses.

    Returns:
        float or None: The calculated pixels_per_foot scale, or None.
        tuple or None: The bounding box of the reference person.
    """
    min_dist = float('inf')
    best_scale = None
    ref_person_box = None

    if not pose_results or pose_results[0].keypoints.shape[0] == 0:
        return None, None

    for person in pose_results[0].keypoints.data:
        person_pixel_height = get_pixel_height_from_pose(person)
        
        if person_pixel_height:
            nose_pt = person[0][:2]
            avg_ankle_pt_y = (person[15][1] + person[16][1]) / 2
            person_center_x = nose_pt[0]
            person_center_y = (nose_pt[1] + avg_ankle_pt_y) / 2
            
            dist = math.sqrt((steel_center[0] - person_center_x)**2 + (steel_center[1] - person_center_y)**2)

            if dist < min_dist:
                min_dist = dist
                best_scale = person_pixel_height / REFERENCE_PERSON_HEIGHT_FEET
                for box in pose_results[0].boxes.xyxy:
                    if box[0] < nose_pt[0] < box[2] and box[1] < nose_pt[1] < box[3]:
                        ref_person_box = box
                        break

    return best_scale, ref_person_box


def main():
    video_path = r"C:\Users\STUDENT\Downloads\RR constructions centring material-20250704T045212Z-1-003\RR constructions centring material\Centring sheet\Copy of VID_20250605_111315.mp4"
    output_path = r"D:\steel\output\steel_measurement_output.mp4"

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
            if frame_num % 30 == 0:
                print(f"[INFO] Processing frame {frame_num}...")

            pose_results = pose_model(frame, verbose=False)

            steel_results = steel_model(frame, conf=0.4, verbose=False)

            annotated_frame = frame.copy()
            
            if steel_results[0].masks:
                polygons = steel_results[0].masks.xy
                for poly in polygons:
                    contour = np.array(poly, dtype=np.int32)
                    
                    rect = cv2.minAreaRect(contour)
                    (w_px, h_px) = rect[1]
                    center_px = tuple(np.array(rect[0]).astype(int))
                    box_points = np.array(cv2.boxPoints(rect)).astype(int)

                    cv2.drawContours(annotated_frame, [box_points], 0, (0, 255, 0), 2)
                    
                    pixels_per_foot, ref_person_box = find_closest_person_scale(center_px, pose_results)
                    
                    dim_text = f"W:{w_px:.1f}px, H:{h_px:.1f}px" # Default text
                    
                    if pixels_per_foot and pixels_per_foot > 0:
                        # Ensure we don't divide by zero
                        w_ft = w_px / pixels_per_foot
                        h_ft = h_px / pixels_per_foot
                        
                        width_ft = max(w_ft, h_ft)
                        height_ft = min(w_ft, h_ft)

                        dim_text = f"W: {width_ft:.2f} ft, H: {height_ft:.2f} ft"
                        
                        if ref_person_box is not None:
                            person_center_x = int((ref_person_box[0] + ref_person_box[2]) / 2)
                            person_center_y = int((ref_person_box[1] + ref_person_box[3]) / 2)
                            cv2.line(annotated_frame, center_px, (person_center_x, person_center_y), (255, 255, 0), 2)
                            cv2.rectangle(annotated_frame, (int(ref_person_box[0]), int(ref_person_box[1])), (int(ref_person_box[2]), int(ref_person_box[3])), (255, 255, 0), 2)
                            cv2.putText(annotated_frame, "Reference", (int(ref_person_box[0]), int(ref_person_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


                    cv2.putText(annotated_frame, dim_text, (center_px[0] - 60, center_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            out.write(annotated_frame)
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