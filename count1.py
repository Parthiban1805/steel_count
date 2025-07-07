import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- Constants ---
# When a sheet's overlap with the lorry zone exceeds this, it's counted.
IOU_ENTRY_THRESHOLD = 0.5
# If the gap between two detected boxes is less than this many pixels, merge them.
BOX_MERGE_PROXIMITY_THRESHOLD = 60 


def get_centroid(box):
    """Get center point of a bounding box [x1,y1,x2,y2]"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

# --- NEW FUNCTION TO MERGE FRAGMENTED DETECTIONS ---
def merge_proximal_boxes(boxes, threshold):
    """
    Merges bounding boxes that are very close to each other.
    This fixes the issue of a single object being detected as multiple fragments.
    """
    if not boxes:
        return []

    # Convert to a list of lists for manipulation
    boxes = [list(b) for b in boxes]

    while True:
        merged_in_pass = False
        i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                box1 = boxes[i]
                box2 = boxes[j]

                # Check proximity: calculate the gap between boxes
                # A negative gap means they overlap
                gap_x = max(box1[0], box2[0]) - min(box1[2], box2[2])
                gap_y = max(box1[1], box2[1]) - min(box1[3], box2[3])

                if gap_x < threshold and gap_y < threshold:
                    # Merge box2 into box1
                    boxes[i][0] = min(box1[0], box2[0])
                    boxes[i][1] = min(box1[1], box2[1])
                    boxes[i][2] = max(box1[2], box2[2])
                    boxes[i][3] = max(box1[3], box2[3])

                    # Remove box2 and mark that a merge happened
                    boxes.pop(j)
                    merged_in_pass = True
                    # Do not increment j, as the list has shifted
                else:
                    j += 1
            i += 1

        if not merged_in_pass:
            # If no merges happened in a full pass, we are done
            break
            
    # Convert back to a list of numpy arrays for consistency
    return [np.array(b) for b in boxes]

# (The cluster_detections function for the lorry is still useful and remains unchanged)
def cluster_detections(detections, iou_threshold=0.3):
    """Group similar detections together"""
    if not detections:
        return []
    clusters, used = [], [False] * len(detections)
    for i, det in enumerate(detections):
        if used[i]: continue
        cluster, used[i] = [det], True
        for j, other_det in enumerate(detections[i+1:], i+1):
            if used[j]: continue
            if calculate_iou(det, other_det) > iou_threshold:
                cluster.append(other_det); used[j] = True
        clusters.append(cluster)
    return clusters

def find_stable_lorry_bounding_box(video_path, lorry_model, sample_frames=50, frame_interval=30):
    """Finds stable lorry bounding box (unchanged)."""
    print("[INFO] Analyzing video to detect lorry position...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"[INFO] Video has {total_frames} frames at {fps} FPS")
    all_detections, sample_info = [], []
    for i in range(0, min(sample_frames * frame_interval, total_frames), frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ret, frame = cap.read()
        if not ret: continue
        results = lorry_model(frame, imgsz=640, conf=0.4, verbose=False)[0]
        frame_detections = []
        if results.boxes:
            for box in results.boxes:
                class_name = results.names[int(box.cls)]
                if class_name in ['truck', 'bus', 'car'] and float(box.conf) > 0.4:
                    bbox = box.xyxy[0].cpu().numpy()
                    if (bbox[2] - bbox[0]) > 100 and (bbox[3] - bbox[1]) > 80:
                        frame_detections.append({'bbox': bbox})
        all_detections.extend(frame_detections)
        if len(frame_detections) > 0: print(f"[INFO] Frame {i}: Found {len(frame_detections)} potential lorry(s)")
    cap.release()
    if not all_detections: print("[ERROR] No lorry detections found"); return None, []
    detection_boxes = [det['bbox'] for det in all_detections]
    clusters = cluster_detections(detection_boxes, iou_threshold=0.4)
    if not clusters: print("[ERROR] No stable lorry clusters found"); return None, []
    largest_cluster = max(clusters, key=len)
    print(f"[INFO] Found {len(clusters)} clusters, largest has {len(largest_cluster)} detections")
    avg_bbox = np.mean(np.array(largest_cluster), axis=0)
    padding = 30
    avg_bbox = [avg_bbox[0]-padding, avg_bbox[1]-padding, avg_bbox[2]+padding, avg_bbox[3]+padding]
    avg_bbox = [max(0, avg_bbox[0]), max(0, avg_bbox[1]), avg_bbox[2], avg_bbox[3]]
    print(f"[INFO] Stable lorry zone: [{avg_bbox[0]:.0f}, {avg_bbox[1]:.0f}, {avg_bbox[2]:.0f}, {avg_bbox[3]:.0f}]")
    return avg_bbox, sample_info

# (MetalSheetTracker class remains unchanged as its logic is now sound with proper input)
class MetalSheetTracker:
    def __init__(self, iou_match_threshold=0.3, max_missing_frames=30):
        self.next_id = 0
        self.tracks = {}
        self.counted_sheets = set()
        self.frame_count = 0
        self.iou_match_threshold = iou_match_threshold
        self.max_missing_frames = max_missing_frames
    def update(self, detected_boxes, lorry_box):
        self.frame_count += 1
        if not detected_boxes: self._remove_stale_tracks(); return
        matched_track_ids, used_detection_indices = set(), set()
        for track_id, track_data in self.tracks.items():
            best_match_iou, best_match_idx = 0, -1
            for i, det_box in enumerate(detected_boxes):
                if i in used_detection_indices: continue
                iou = calculate_iou(track_data['box'], det_box)
                if iou > best_match_iou: best_match_iou, best_match_idx = iou, i
            if best_match_iou > self.iou_match_threshold:
                det_box = detected_boxes[best_match_idx]
                prev_iou = track_data['prev_iou']
                current_iou = calculate_iou(det_box, lorry_box)
                if prev_iou < IOU_ENTRY_THRESHOLD and current_iou >= IOU_ENTRY_THRESHOLD and track_id not in self.counted_sheets:
                    self.counted_sheets.add(track_id)
                    print(f"[COUNT] Sheet #{self.get_count()} (ID: {track_id}) entered lorry at frame {self.frame_count}")
                self.tracks[track_id].update({'box': det_box, 'last_seen': self.frame_count, 'prev_iou': current_iou})
                matched_track_ids.add(track_id); used_detection_indices.add(best_match_idx)
        for i, det_box in enumerate(detected_boxes):
            if i not in used_detection_indices:
                iou_with_lorry = calculate_iou(det_box, lorry_box)
                self.tracks[self.next_id] = {'box': det_box, 'last_seen': self.frame_count, 'prev_iou': iou_with_lorry}
                if iou_with_lorry >= IOU_ENTRY_THRESHOLD and self.next_id not in self.counted_sheets:
                    self.counted_sheets.add(self.next_id)
                    print(f"[COUNT] New sheet #{self.get_count()} (ID: {self.next_id}) detected inside at frame {self.frame_count}")
                self.next_id += 1
        self._remove_stale_tracks()
    def _remove_stale_tracks(self):
        to_remove = [tid for tid, data in self.tracks.items() if self.frame_count - data['last_seen'] > self.max_missing_frames]
        for tid in to_remove: del self.tracks[tid]
    def get_count(self): return len(self.counted_sheets)


def main():
    # Use the new video path and model path from your latest code
    video_path = r"C:\Users\STUDENT\Downloads\RR constructions centring material-20250704T045212Z-1-005\RR constructions centring material\Centring span\Copy of VID_20250605_115251.mp4"
    output_path = r"D:\steel\output\count_merged_boxes.mp4"
    
    print("="*60)
    print("AUTOMATED LORRY & METAL SHEET COUNTER (with Box Merging)")
    print("="*60)
    
    print("[INFO] Loading YOLO models...")
    try:
        metal_model = YOLO(r"D:\steel\runs\detect\steel_detector_large_span\weights\best.pt")
        lorry_model = YOLO("yolov8l.pt")
        print("[INFO] Models loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load models - {e}"); return
    
    lorry_bbox, _ = find_stable_lorry_bounding_box(video_path, lorry_model, sample_frames=30, frame_interval=30)
    if lorry_bbox is None: print("ERROR: Could not detect stable lorry position. Exiting."); return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print("ERROR: Cannot open video file"); return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_w, out_h = 1280, int(h * 1280 / w)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    
    tracker = MetalSheetTracker(iou_match_threshold=0.2, max_missing_frames=20)
    frame_num = 0
    print("\n[INFO] Starting video processing with fixed lorry zone...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_num += 1
            if frame_num % fps == 0:
                progress = (frame_num / total_frames) * 100
                print(f"[PROGRESS] {progress:.1f}% | Frame: {frame_num}/{total_frames} | Count: {tracker.get_count()}")
            
            # --- MODIFIED DETECTION AND TRACKING PIPELINE ---
            # 1. Detect all potential metal sheet fragments
            metal_results = metal_model(frame, imgsz=640, conf=0.4, verbose=False)[0]
            raw_metal_boxes = [box.xyxy[0].cpu().numpy() for box in metal_results.boxes] if metal_results.boxes else []
            
            # 2. Pre-process: Merge fragmented boxes into single, coherent boxes
            merged_metal_boxes = merge_proximal_boxes(raw_metal_boxes, BOX_MERGE_PROXIMITY_THRESHOLD)

            if len(raw_metal_boxes) > len(merged_metal_boxes) > 0:
                 print(f"[MERGE] Frame {frame_num}: Merged {len(raw_metal_boxes)} raw boxes into {len(merged_metal_boxes)} coherent boxes.")

            # 3. Update tracker with the clean, merged boxes
            tracker.update(merged_metal_boxes, lorry_bbox)
            
            # --- Drawing Logic (remains the same, but now draws more accurate tracks) ---
            annotated_frame = frame.copy()
            x1, y1, x2, y2 = map(int, lorry_bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
            cv2.putText(annotated_frame, "LORRY ZONE", (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            for track_id, track_data in tracker.tracks.items():
                box = track_data['box']
                bx1, by1, bx2, by2 = map(int, box)
                color = (0, 0, 255) if track_id in tracker.counted_sheets else (0, 255, 0)
                label = "METAL (IN LORRY)" if track_id in tracker.counted_sheets else "METAL (OUTSIDE)"
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(annotated_frame, label, (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cx, cy = map(int, get_centroid(box))
                cv2.circle(annotated_frame, (cx, cy), 6, color, -1)
            
            count_text = f"Sheets Entered Lorry: {tracker.get_count()}"
            text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.rectangle(annotated_frame, (5, 5), (text_size[0] + 20, text_size[1] + 25), (255, 255, 255), -1)
            cv2.putText(annotated_frame, count_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"Frame: {frame_num}/{total_frames}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out.write(cv2.resize(annotated_frame, (out_w, out_h)))
            cv2.imshow('AI Lorry Counter (Box Merging) - Press Q to exit', cv2.resize(annotated_frame, (960, 540)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}"); import traceback; traceback.print_exc()
    
    finally:
        cap.release(); out.release(); cv2.destroyAllWindows()
        print("\n" + "="*60 + "\nPROCESSING COMPLETE\n" + "="*60)
        print(f"Final count: {tracker.get_count()} metal sheets entered the lorry")
        print(f"Output video saved to: {output_path}\n" + "="*60)

if __name__ == "__main__":
    main()