import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- Helper Functions (Moved to top level for general use) ---

def get_centroid(box):
    """Get center point of a bounding box [x1,y1,x2,y2]"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def point_in_box(point, box):
    """Check if a point (x,y) is inside a bounding box [x1,y1,x2,y2]"""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# MODIFIED: Moved this function to be globally accessible
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

# NEW: Function to merge overlapping bounding boxes for a single large object
def merge_overlapping_boxes(boxes, iou_threshold=0.05):
    """
    Merge overlapping bounding boxes into single encompassing boxes.
    A low IoU threshold is used to merge boxes that are just adjacent or slightly overlapping.
    """
    if not boxes:
        return []

    # Start with a list of boxes to be processed
    merged_boxes = []
    
    # Use a boolean array to mark boxes that have been merged
    used = [False] * len(boxes)

    for i, box1 in enumerate(boxes):
        if used[i]:
            continue

        # This box is the start of a new potential merged box
        current_cluster = [box1]
        used[i] = True

        for j, box2 in enumerate(boxes[i+1:], start=i+1):
            if used[j]:
                continue
            
            # Check if box2 overlaps with any box in the current cluster
            for cluster_box in current_cluster:
                if calculate_iou(cluster_box, box2) > iou_threshold:
                    current_cluster.append(box2)
                    used[j] = True
                    break # Move to the next box once merged

        # After checking all other boxes, create the final merged box for the cluster
        if len(current_cluster) > 1:
            x_coords = []
            y_coords = []
            for b in current_cluster:
                x_coords.extend([b[0], b[2]])
                y_coords.extend([b[1], b[3]])
            
            merged_box = np.array([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])
            merged_boxes.append(merged_box)
        else:
            # Not a cluster, just a single box
            merged_boxes.append(box1)
            
    return merged_boxes


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
    Sample frames from video to find stable lorry bounding box
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
    sample_info = []
    
    for i in range(0, min(sample_frames * frame_interval, total_frames), frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        results = lorry_model(frame, imgsz=640, conf=0.4, verbose=False)[0]
        
        frame_detections = []
        if results.boxes:
            for box in results.boxes:
                class_name = results.names[int(box.cls)]
                confidence = float(box.conf)
                
                if class_name in ['truck', 'bus', 'car'] and confidence > 0.4:
                    bbox = box.xyxy[0].cpu().numpy()
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    if width > 100 and height > 80:
                        frame_detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'class': class_name,
                            'frame': i
                        })
        
        all_detections.extend(frame_detections)
        sample_info.append({
            'frame': i,
            'detections': len(frame_detections)
        })
        
        if len(frame_detections) > 0:
            print(f"[INFO] Frame {i}: Found {len(frame_detections)} potential lorry(s)")
    
    cap.release()
    
    if not all_detections:
        print("[ERROR] No lorry detections found in sampled frames")
        return None, sample_info
    
    print(f"[INFO] Total detections found: {len(all_detections)}")
    
    detection_boxes = [det['bbox'] for det in all_detections]
    clusters = cluster_detections(detection_boxes, iou_threshold=0.4)
    
    if not clusters:
        print("[ERROR] No stable lorry clusters found")
        return None, sample_info
        
    largest_cluster = max(clusters, key=len)
    
    print(f"[INFO] Found {len(clusters)} clusters, largest has {len(largest_cluster)} detections")
    
    cluster_array = np.array(largest_cluster)
    avg_bbox = np.mean(cluster_array, axis=0)
    
    padding = 30
    avg_bbox[0] = max(0, avg_bbox[0] - padding)
    avg_bbox[1] = max(0, avg_bbox[1] - padding)
    avg_bbox[2] += padding
    avg_bbox[3] += padding
    
    print(f"[INFO] Stable lorry zone calculated: [{avg_bbox[0]:.0f}, {avg_bbox[1]:.0f}, {avg_bbox[2]:.0f}, {avg_bbox[3]:.0f}]")
    
    return avg_bbox, sample_info

class MetalSheetTracker:
    def __init__(self, distance_threshold=50, max_missing_frames=30):
        self.next_id = 0
        self.tracks = {}
        self.counted_sheets = set()
        self.frame_count = 0
        self.distance_threshold = distance_threshold
        self.max_missing_frames = max_missing_frames
        
    def update(self, metal_centroids, lorry_box):
        self.frame_count += 1
        matched_tracks = set()
        
        for centroid in metal_centroids:
            best_match_id = None
            best_distance = float('inf')
            
            for track_id, track_data in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                prev_center = track_data['center']
                distance = np.sqrt((centroid[0] - prev_center[0])**2 + (centroid[1] - prev_center[1])**2)
                
                if distance < best_distance and distance < self.distance_threshold:
                    best_distance = distance
                    best_match_id = track_id
            
            if best_match_id is not None:
                was_inside = self.tracks[best_match_id]['inside_lorry']
                is_inside = point_in_box(centroid, lorry_box) if lorry_box is not None else False
                
                self.tracks[best_match_id].update({
                    'center': centroid,
                    'last_seen': self.frame_count,
                    'inside_lorry': is_inside
                })
                
                if not was_inside and is_inside and best_match_id not in self.counted_sheets:
                    self.counted_sheets.add(best_match_id)
                    print(f"[INFO] Sheet #{len(self.counted_sheets)} entered lorry at frame {self.frame_count}")
                
                matched_tracks.add(best_match_id)
            else:
                is_inside = point_in_box(centroid, lorry_box) if lorry_box is not None else False
                self.tracks[self.next_id] = {
                    'center': centroid,
                    'last_seen': self.frame_count,
                    'inside_lorry': is_inside
                }
                matched_tracks.add(self.next_id)
                self.next_id += 1
        
        tracks_to_remove = [track_id for track_id, data in self.tracks.items() if self.frame_count - data['last_seen'] > self.max_missing_frames]
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_count(self):
        return len(self.counted_sheets)

def main():
    video_path = r"C:\Users\STUDENT\Downloads\RR constructions centring material-20250704T045212Z-1-005\RR constructions centring material\Centring span\Copy of VID_20250605_115251.mp4"
    output_path = r"D:\steel\output\count1_merged.mp4" # Changed output name
    
    print("="*60)
    print("AUTOMATED LORRY DETECTION & METAL SHEET COUNTER")
    print("="*60)
    
    try:
        metal_model = YOLO(r"C:\Users\STUDENT\Downloads\best_span.pt")
        lorry_model = YOLO("yolov8l.pt")
        print("[INFO] Models loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load models - {e}")
        return
    
    lorry_bbox, sample_info = find_stable_lorry_bounding_box(
        video_path, lorry_model, sample_frames=30, frame_interval=30
    )
    
    if lorry_bbox is None:
        print("ERROR: Could not detect stable lorry position in video")
        return
        
    total_detections = sum(info['detections'] for info in sample_info)
    frames_with_detections = len([info for info in sample_info if info['detections'] > 0])
    
    print(f"[INFO] Detection Statistics:\n  - Frames analyzed: {len(sample_info)}\n  - Frames with detections: {frames_with_detections}\n  - Total detections: {total_detections}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video file")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output_width = 1280
    output_height = int(height * output_width / width)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        print("ERROR: Cannot create output video file")
        return
        
    tracker = MetalSheetTracker(distance_threshold=150, max_missing_frames=20) # Increased distance threshold for larger merged objects
    
    frame_num = 0
    print("\n[INFO] Starting video processing with fixed lorry zone...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            if frame_num % fps == 0:
                progress = (frame_num / total_frames) * 100
                print(f"[INFO] Progress: {progress:.1f}% | Frame: {frame_num}/{total_frames} | Count: {tracker.get_count()}")
            
            # 1. Detect metal sheets
            metal_results = metal_model(frame, imgsz=640, conf=0.4, verbose=False)[0]
            raw_metal_boxes = []
            if metal_results.boxes:
                raw_metal_boxes = [box.xyxy[0].cpu().numpy() for box in metal_results.boxes]
            
            # 2. NEW: Merge overlapping boxes to treat large sheets as one object
            # You can tune the iou_threshold. 0.0 means any touching boxes will merge.
            merged_metal_boxes = merge_overlapping_boxes(raw_metal_boxes, iou_threshold=0.05)

            # 3. MODIFIED: Get centroids of the MERGED metal sheets
            metal_centroids = [get_centroid(box) for box in merged_metal_boxes]
            
            # 4. Update tracker with the consolidated list of objects
            tracker.update(metal_centroids, lorry_bbox)
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # Draw fixed lorry bounding box
            x1, y1, x2, y2 = map(int, lorry_bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
            cv2.putText(annotated_frame, "LORRY ZONE (AI DETECTED)", (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # MODIFIED: Draw the MERGED metal sheet detections
            for i, box in enumerate(merged_metal_boxes):
                bx1, by1, bx2, by2 = map(int, box)
                centroid = metal_centroids[i]
                
                inside_lorry = point_in_box(centroid, lorry_bbox)
                color = (0, 0, 255) if inside_lorry else (0, 255, 0)
                label = "METAL (IN LORRY)" if inside_lorry else "METAL (OUTSIDE)"
                
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(annotated_frame, label, (bx1, by1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cx, cy = map(int, centroid)
                cv2.circle(annotated_frame, (cx, cy), 6, color, -1)
            
            # Display count and info
            count_text = f"Sheets Entered Lorry: {tracker.get_count()}"
            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.rectangle(annotated_frame, (5, 5), (text_size[0] + 20, text_size[1] + 25), (255, 255, 255), -1)
            cv2.putText(annotated_frame, count_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            progress = (frame_num / total_frames) * 100
            cv2.putText(annotated_frame, f"Frame: {frame_num}/{total_frames}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Progress: {progress:.1f}%", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Write and display
            output_frame = cv2.resize(annotated_frame, (output_width, output_height))
            out.write(output_frame)
            
            display_frame = cv2.resize(output_frame, (960, int(output_height * 960 / output_width)))
            cv2.imshow('AI Lorry Detection + Metal Sheet Counter - Press Q to exit', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] User requested exit")
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        final_count = tracker.get_count()
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Final count: {final_count} metal sheets entered the lorry")
        print(f"Total frames processed: {frame_num}")
        print(f"Output video saved to: {output_path}")
        print("="*60)

if __name__ == "__main__":
    main()