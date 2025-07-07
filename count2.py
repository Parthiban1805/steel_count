import cv2
import numpy as np
from ultralytics import YOLO
import os

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
    Sample frames from video to find stable lorry bounding box
    
    Args:
        video_path: Path to video file
        lorry_model: YOLO model for lorry detection
        sample_frames: Number of frames to sample
        frame_interval: Interval between sampled frames
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
    
    # Sample frames at regular intervals
    for i in range(0, min(sample_frames * frame_interval, total_frames), frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # Detect lorries with higher confidence
        results = lorry_model(frame, imgsz=640, conf=0.4, verbose=False)[0]
        
        frame_detections = []
        if results.boxes:
            for box in results.boxes:
                class_name = results.names[int(box.cls)]
                confidence = float(box.conf)
                
                # Look for trucks, buses, and sometimes cars (large vehicles)
                if class_name in ['truck', 'bus', 'car'] and confidence > 0.4:
                    bbox = box.xyxy[0].cpu().numpy()
                    # Filter out very small detections (likely not lorries)
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    if width > 100 and height > 80:  # Minimum size filter
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
    
    # Extract just the bounding boxes for clustering
    detection_boxes = [det['bbox'] for det in all_detections]
    
    # Cluster similar detections
    clusters = cluster_detections(detection_boxes, iou_threshold=0.4)
    
    if not clusters:
        print("[ERROR] No stable lorry clusters found")
        return None, sample_info
    
    # Find the largest cluster (most consistent detections)
    largest_cluster = max(clusters, key=len)
    
    print(f"[INFO] Found {len(clusters)} clusters, largest has {len(largest_cluster)} detections")
    
    # Calculate average bounding box from the largest cluster
    cluster_array = np.array(largest_cluster)
    avg_bbox = np.mean(cluster_array, axis=0)
    
    # Add some padding for safety
    padding = 30
    avg_bbox[0] -= padding  # x1
    avg_bbox[1] -= padding  # y1
    avg_bbox[2] += padding  # x2
    avg_bbox[3] += padding  # y2
    
    # Ensure bounds are within frame
    avg_bbox[0] = max(0, avg_bbox[0])
    avg_bbox[1] = max(0, avg_bbox[1])
    
    print(f"[INFO] Stable lorry zone calculated: [{avg_bbox[0]:.0f}, {avg_bbox[1]:.0f}, {avg_bbox[2]:.0f}, {avg_bbox[3]:.0f}]")
    
    return avg_bbox, sample_info

class MetalSheetTracker:
    def __init__(self, distance_threshold=50, max_missing_frames=30):
        self.next_id = 0
        self.tracks = {}  # id -> {'center': (x,y), 'last_seen': frame_num, 'inside_lorry': bool}
        self.counted_sheets = set()  # IDs of sheets that entered lorry
        self.frame_count = 0
        self.distance_threshold = distance_threshold
        self.max_missing_frames = max_missing_frames
        
    def update(self, metal_centroids, lorry_box):
        """Update tracking with new detections"""
        self.frame_count += 1
        matched_tracks = set()
        
        # Match current detections to existing tracks
        for centroid in metal_centroids:
            best_match_id = None
            best_distance = float('inf')
            
            # Find closest existing track
            for track_id, track_data in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                    
                prev_center = track_data['center']
                distance = np.sqrt((centroid[0] - prev_center[0])**2 + 
                                 (centroid[1] - prev_center[1])**2)
                
                if distance < best_distance and distance < self.distance_threshold:
                    best_distance = distance
                    best_match_id = track_id
            
            # Update existing track or create new one
            if best_match_id is not None:
                # Update existing track
                was_inside = self.tracks[best_match_id]['inside_lorry']
                is_inside = point_in_box(centroid, lorry_box) if lorry_box is not None else False
                
                self.tracks[best_match_id].update({
                    'center': centroid,
                    'last_seen': self.frame_count,
                    'inside_lorry': is_inside
                })
                
                # Count if sheet just entered lorry
                if not was_inside and is_inside and best_match_id not in self.counted_sheets:
                    self.counted_sheets.add(best_match_id)
                    print(f"[INFO] Sheet #{len(self.counted_sheets)} entered lorry at frame {self.frame_count}")
                
                matched_tracks.add(best_match_id)
            else:
                # Create new track
                is_inside = point_in_box(centroid, lorry_box) if lorry_box is not None else False
                self.tracks[self.next_id] = {
                    'center': centroid,
                    'last_seen': self.frame_count,
                    'inside_lorry': is_inside
                }
                matched_tracks.add(self.next_id)
                self.next_id += 1
        
        # Remove old tracks that haven't been seen
        tracks_to_remove = []
        for track_id, track_data in self.tracks.items():
            if self.frame_count - track_data['last_seen'] > self.max_missing_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_count(self):
        """Get total count of sheets that entered lorry"""
        return len(self.counted_sheets)

def main():
    # Configuration - Using the same paths from your code
    video_path = r"C:\Users\STUDENT\Downloads\RR constructions centring material-20250704T045212Z-1-003\RR constructions centring material\Centring sheet\Copy of VID_20250605_111315.mp4"
    output_path = r"D:\steel\output\count1.mp4"
    
    print("="*60)
    print("AUTOMATED LORRY DETECTION & METAL SHEET COUNTER")
    print("="*60)
    
    # Load models
    print("[INFO] Loading YOLO models...")
    try:
        metal_model = YOLO(r"D:\steel\runs\detect\steel_detector_large3\weights\best.pt")
        lorry_model = YOLO("yolov8l.pt")  # Pre-trained COCO model
        print("[INFO] Models loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load models - {e}")
        return
    
    # Find stable lorry bounding box by analyzing multiple frames
    lorry_bbox, sample_info = find_stable_lorry_bounding_box(
        video_path, lorry_model, 
        sample_frames=30,  # Sample 30 frames
        frame_interval=30   # Every 60 frames (every 2 seconds at 30fps)
    )
    
    if lorry_bbox is None:
        print("ERROR: Could not detect stable lorry position in video")
        print("Suggestion: Try manual selection mode or check if lorry is visible in video")
        return
    
    # Show detection statistics
    total_detections = sum(info['detections'] for info in sample_info)
    frames_with_detections = len([info for info in sample_info if info['detections'] > 0])
    
    print(f"[INFO] Detection Statistics:")
    print(f"  - Frames analyzed: {len(sample_info)}")
    print(f"  - Frames with detections: {frames_with_detections}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Average detections per frame: {total_detections/len(sample_info):.2f}")
    
    # Initialize video capture for main processing
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Resize for output (to manage file size)
    output_width = 1280
    output_height = int(height * output_width / width)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        print("ERROR: Cannot create output video file")
        return
    
    # Initialize tracker
    tracker = MetalSheetTracker(distance_threshold=80, max_missing_frames=20)
    
    frame_num = 0
    print("\n[INFO] Starting video processing with fixed lorry zone...")
    print("Press 'q' during playback to stop early")
    print("="*50)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Show progress every second
            if frame_num % fps == 0:
                progress = (frame_num / total_frames) * 100
                print(f"[INFO] Progress: {progress:.1f}% | Frame: {frame_num}/{total_frames} | Count: {tracker.get_count()}")
            
            # Detect metal sheets
            metal_results = metal_model(frame, imgsz=640, conf=0.4, verbose=False)[0]
            metal_boxes = []
            if metal_results.boxes:
                metal_boxes = [box.xyxy[0].cpu().numpy() for box in metal_results.boxes]
            
            # Get centroids of detected metal sheets
            metal_centroids = [get_centroid(box) for box in metal_boxes]
            
            # Update tracker with fixed lorry bbox
            tracker.update(metal_centroids, lorry_bbox)
            
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # Draw fixed lorry bounding box
            x1, y1, x2, y2 = map(int, lorry_bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 4)  # Yellow
            cv2.putText(annotated_frame, "LORRY ZONE (AI DETECTED)", (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Draw metal sheet detections
            for i, box in enumerate(metal_boxes):
                bx1, by1, bx2, by2 = map(int, box)
                centroid = metal_centroids[i]
                
                # Check if metal sheet is inside lorry
                inside_lorry = point_in_box(centroid, lorry_bbox)
                
                if inside_lorry:
                    # Red for sheets inside lorry
                    color = (0, 0, 255)
                    label = "METAL (IN LORRY)"
                else:
                    # Green for sheets outside lorry
                    color = (0, 255, 0)
                    label = "METAL (OUTSIDE)"
                
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(annotated_frame, label, (bx1, by1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw centroid
                cx, cy = map(int, centroid)
                cv2.circle(annotated_frame, (cx, cy), 6, color, -1)
            
            # Display count with better visibility
            count_text = f"Sheets Entered Lorry: {tracker.get_count()}"
            # White background for text
            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.rectangle(annotated_frame, (5, 5), (text_size[0] + 20, text_size[1] + 25), (255, 255, 255), -1)
            cv2.putText(annotated_frame, count_text, (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Display frame number
            frame_text = f"Frame: {frame_num}/{total_frames}"
            cv2.putText(annotated_frame, frame_text, (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Display progress percentage
            progress = (frame_num / total_frames) * 100
            progress_text = f"Progress: {progress:.1f}%"
            cv2.putText(annotated_frame, progress_text, (10, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Resize for output and display
            output_frame = cv2.resize(annotated_frame, (output_width, output_height))
            
            # Write to output video
            out.write(output_frame)
            
            # Display frame (smaller window for performance)
            display_frame = cv2.resize(output_frame, (960, int(output_height * 960 / output_width)))
            cv2.imshow('AI Lorry Detection + Metal Sheet Counter - Press Q to exit', display_frame)
            
            # Exit on 'q' key
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
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        final_count = tracker.get_count()
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Final count: {final_count} metal sheets entered the lorry")
        print(f"Total frames processed: {frame_num}")
        print(f"Lorry zone used: [{lorry_bbox[0]:.0f}, {lorry_bbox[1]:.0f}, {lorry_bbox[2]:.0f}, {lorry_bbox[3]:.0f}]")
        print(f"Output video saved to: {output_path}")
        print("="*60)

if __name__ == "__main__":
    main()