import cv2
import numpy as np
from ultralytics import YOLO, SAM
import os
import time

# --- Helper Functions (from original code) ---

def get_centroid(box):
    """Get center point of a bounding box [x1,y1,x2,y2]"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def point_in_box(point, box):
    """Check if a point (x,y) is inside a bounding box [x1,y1,x2,y2]"""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def cluster_detections(detections, iou_threshold=0.3):
    """Group similar detections together"""
    if not detections:
        return []
    
    clusters = []
    used = [False] * len(detections)
    
    for i, det in enumerate(detections):
        if used[i]:
            continue
            
        cluster = [det]
        used[i] = True
        
        for j, other_det in enumerate(detections[i+1:], i+1):
            if used[j]:
                continue
                
            if calculate_iou(det, other_det) > iou_threshold:
                cluster.append(other_det)
                used[j] = True
        
        clusters.append(cluster)
    
    return clusters

def find_stable_lorry_bounding_box(video_path, lorry_model, sample_frames=50, frame_interval=30):
    """
    Sample frames from video to find stable lorry bounding box.
    This function remains the same.
    """
    print("[INFO] Analyzing video to detect lorry position...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video file")
        return None, []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"[INFO] Video has {total_frames} frames at {fps} FPS")
    
    all_detections = []
    # Sample frames at regular intervals
    for i in range(0, min(sample_frames * frame_interval, total_frames), frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
            
        results = lorry_model(frame, imgsz=640, conf=0.4, verbose=False)[0]
        frame_detections = []
        if results.boxes:
            for box in results.boxes:
                class_name = results.names[int(box.cls)]
                if class_name in ['truck', 'bus', 'car']:
                    bbox = box.xyxy[0].cpu().numpy()
                    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if width > 100 and height > 80:
                        frame_detections.append({'bbox': bbox})
        
        all_detections.extend([det['bbox'] for det in frame_detections])
        if len(frame_detections) > 0:
            print(f"[INFO] Frame {i}: Found {len(frame_detections)} potential lorry(s)")

    cap.release()
    if not all_detections:
        print("[ERROR] No lorry detections found in sampled frames")
        return None, []

    print(f"[INFO] Total potential lorry detections: {len(all_detections)}")
    clusters = cluster_detections(all_detections, iou_threshold=0.4)
    if not clusters:
        print("[ERROR] No stable lorry clusters found")
        return None, []

    largest_cluster = max(clusters, key=len)
    print(f"[INFO] Found {len(clusters)} clusters, largest has {len(largest_cluster)} detections")
    
    avg_bbox = np.mean(np.array(largest_cluster), axis=0)
    padding = 30
    avg_bbox = [
        max(0, avg_bbox[0] - padding), max(0, avg_bbox[1] - padding),
        avg_bbox[2] + padding, avg_bbox[3] + padding
    ]
    print(f"[INFO] Stable lorry zone calculated: [{avg_bbox[0]:.0f}, {avg_bbox[1]:.0f}, {avg_bbox[2]:.0f}, {avg_bbox[3]:.0f}]")
    return avg_bbox


# --- NEW FUNCTION TO INTEGRATE SAM ---

def refine_detections_with_sam(frame, yolo_results, sam_model):
    """
    Refine YOLO bounding boxes using SAM to get precise segmentations
    and individual object detections.
    """
    refined_detections = []
    
    if not yolo_results.boxes:
        return refined_detections

    # Get all coarse bounding boxes from YOLO
    yolo_boxes = yolo_results.boxes.xyxy.cpu().numpy()

    if len(yolo_boxes) == 0:
        return refined_detections

    # Run SAM on the frame with all YOLO boxes as prompts
    # This is more efficient than running SAM for each box individually
    sam_results = sam_model(frame, bboxes=yolo_boxes, verbose=False)

    if not sam_results or not sam_results[0].masks:
        return refined_detections # Return empty if SAM fails

    # Process each mask generated by SAM
    for mask_data in sam_results[0].masks.data:
        # Convert mask tensor to a binary image
        mask = mask_data.cpu().numpy().astype(np.uint8)
        
        # Find contours for each disconnected object in the mask
        # This is the key step to separate multiple sheets from one YOLO box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter out very small, noisy contours
            if cv2.contourArea(contour) < 100:
                continue

            # Get the bounding box of the individual segmented object
            x, y, w, h = cv2.boundingRect(contour)
            refined_box = [x, y, x + w, y + h]
            
            # Get the precise centroid of the segmented object
            M = cv2.moments(contour)
            if M["m00"] != 0:
                centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                refined_detections.append({
                    'bbox': refined_box,
                    'centroid': centroid,
                    'contour': contour # Store contour for drawing
                })

    return refined_detections


# --- Tracker Class (remains the same) ---

class MetalSheetTracker:
    def __init__(self, distance_threshold=80, max_missing_frames=30):
        self.next_id = 0
        self.tracks = {}  # id -> {'center': (x,y), 'last_seen': frame_num, 'inside_lorry': bool}
        self.counted_sheets = set()  # IDs of sheets that entered lorry
        self.frame_count = 0
        self.distance_threshold = distance_threshold
        self.max_missing_frames = max_missing_frames
        
    def update(self, metal_detections, lorry_box):
        """Update tracking with new detections (list of centroids)"""
        self.frame_count += 1
        metal_centroids = [det['centroid'] for det in metal_detections]
        matched_tracks = set()
        
        for centroid in metal_centroids:
            best_match_id, best_distance = None, float('inf')
            
            for track_id, track_data in self.tracks.items():
                if track_id in matched_tracks: continue
                dist = np.linalg.norm(np.array(centroid) - np.array(track_data['center']))
                if dist < best_distance and dist < self.distance_threshold:
                    best_distance, best_match_id = dist, track_id
            
            if best_match_id is not None:
                was_inside = self.tracks[best_match_id]['inside_lorry']
                is_inside = point_in_box(centroid, lorry_box) if lorry_box is not None else False
                self.tracks[best_match_id].update({'center': centroid, 'last_seen': self.frame_count, 'inside_lorry': is_inside})
                if not was_inside and is_inside and best_match_id not in self.counted_sheets:
                    self.counted_sheets.add(best_match_id)
                    print(f"[COUNT] Sheet #{len(self.counted_sheets)} entered lorry at frame {self.frame_count}")
                matched_tracks.add(best_match_id)
            else:
                is_inside = point_in_box(centroid, lorry_box) if lorry_box is not None else False
                self.tracks[self.next_id] = {'center': centroid, 'last_seen': self.frame_count, 'inside_lorry': is_inside}
                self.next_id += 1
        
        # Remove old tracks
        tracks_to_remove = [tid for tid, tdata in self.tracks.items() if self.frame_count - tdata['last_seen'] > self.max_missing_frames]
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_count(self):
        return len(self.counted_sheets)

# --- Main Execution ---

def main():
    # --- Configuration ---
    video_path = r"C:\Users\STUDENT\Downloads\Copy of VID_20250605_115732.mp4"
    output_path = r"D:\steel\output\count_sam_refined.mp4"
    yolo_metal_model_path = r"D:\steel\runs\detect\steel_detector_rtdetr_l\weights\best.pt"
    sam_model_path = "sam_b.pt" # Standard SAM model, e.g., 'sam_b.pt', 'sam_l.pt', or your 'sam2.1_b.pt'
    
    print("="*60)
    print("AI METAL SHEET COUNTER with SAM REFINEMENT")
    print("="*60)
    
    # --- Load Models ---
    print("[INFO] Loading YOLO and SAM models...")
    try:
        metal_model = YOLO(yolo_metal_model_path)
        lorry_model = YOLO("yolov8l.pt")
        sam_model = SAM(sam_model_path)
        print("[INFO] Models loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load models - {e}")
        return
        
    # --- Step 1: Find Stable Lorry Bounding Box ---
    lorry_bbox = find_stable_lorry_bounding_box(
        video_path, lorry_model, 
        sample_frames=30, frame_interval=30
    )
    if lorry_bbox is None:
        print("ERROR: Could not detect a stable lorry position. Exiting.")
        return

    # --- Step 2: Main Video Processing ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_width, output_height = 1280, int(height * 1280 / width)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_width, output_height))

    tracker = MetalSheetTracker(distance_threshold=100, max_missing_frames=25)
    
    print("\n[INFO] Starting video processing with SAM refinement...")
    print("[WARNING] SAM processing is computationally intensive and will be slow.")
    print("="*50)
    
    frame_num = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_num += 1
            start_time = time.time()
            
            # --- Detection and Refinement Pipeline ---
            # 1. Coarse detection with YOLO
            metal_results = metal_model(frame, imgsz=640, conf=0.4, verbose=False)[0]
            
            # 2. Refine detections with SAM
            refined_detections = refine_detections_with_sam(frame, metal_results, sam_model)

            # 3. Update tracker with refined, individual detections
            tracker.update(refined_detections, lorry_bbox)
            
            end_time = time.time()
            processing_fps = 1 / (end_time - start_time)

            # --- Drawing and Visualization ---
            annotated_frame = frame.copy()
            
            # Draw Lorry Zone
            x1, y1, x2, y2 = map(int, lorry_bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(annotated_frame, "LORRY ZONE", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Draw refined metal sheet detections
            for det in refined_detections:
                centroid = det['centroid']
                is_inside = point_in_box(centroid, lorry_bbox)
                color = (0, 0, 255) if is_inside else (0, 255, 0)
                
                # Draw the precise contour from SAM
                cv2.drawContours(annotated_frame, [det['contour']], -1, color, 2)
                
                # Draw centroid
                cv2.circle(annotated_frame, centroid, 5, color, -1)
            
            # Display Count and Info
            count_text = f"Sheets Entered: {tracker.get_count()}"
            cv2.rectangle(annotated_frame, (5, 5), (450, 120), (0, 0, 0, 0.6), -1)
            cv2.putText(annotated_frame, count_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(annotated_frame, f"Frame: {frame_num}/{total_frames}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"FPS: {processing_fps:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            output_frame = cv2.resize(annotated_frame, (output_width, output_height))
            out.write(output_frame)
            cv2.imshow('AI Metal Sheet Counter (SAM Refined) - Press Q to exit', output_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] User requested exit")
                break
    
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print(f"Final Count: {tracker.get_count()} sheets entered the lorry")
        print(f"Output video saved to: {output_path}")
        print("="*60)

if __name__ == "__main__":
    main()