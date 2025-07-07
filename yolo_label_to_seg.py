import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import os

# --- 1. CONFIGURATION ---
# Update these paths to match your project structure

# Path to the downloaded SAM model checkpoint
SAM_CHECKPOINT = "models/sam_vit_h_4b8939.pth" 
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Base directory containing your 'train' and 'val' image folders
BASE_IMAGE_DIR = "yolo_data/images" 

# Base directory containing your 'train' and 'val' YOLO detection label folders
BASE_DETECTION_LABEL_DIR = "yolo_data/labels" 

# Directory where the new segmentation labels will be saved
# The script will create 'train' and 'val' subfolders inside this directory.
OUTPUT_SEG_DIR = "labels_segmentation"

# List of subsets to process
SUBSETS = ['train', 'val']

# --- 2. LOAD THE SAM MODEL (runs only once) ---
print("Loading SAM model... This might take a moment.")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)
print(f"Model loaded successfully to {DEVICE}.")


# --- 3. PROCESSING FUNCTION ---
def process_subset(subset_name):
    """
    Processes a single subset (e.g., 'train' or 'val') to generate segmentation masks.
    """
    print(f"\n--- Processing '{subset_name}' subset ---")
    
    # Define paths for the current subset
    image_dir_subset = os.path.join(BASE_IMAGE_DIR, subset_name)
    label_dir_subset = os.path.join(BASE_DETECTION_LABEL_DIR, subset_name)
    output_dir_subset = os.path.join(OUTPUT_SEG_DIR, subset_name)

    # Create the output directory for the subset if it doesn't exist
    os.makedirs(output_dir_subset, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir_subset) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Warning: No images found in {image_dir_subset}")
        return

    for i, image_filename in enumerate(image_files):
        # Construct paths for the current image and its labels
        image_path = os.path.join(image_dir_subset, image_filename)
        label_filename = os.path.splitext(image_filename)[0] + ".txt"
        label_path = os.path.join(label_dir_subset, label_filename)
        output_path = os.path.join(output_dir_subset, label_filename)

        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {image_filename}, skipping.")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_filename}, skipping.")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape

        # Set the image for the predictor once per image
        predictor.set_image(image_rgb)
        
        with open(label_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) < 5: continue # Skip malformed lines
                
                class_id = parts[0]
                x_center, y_center, width, height = map(float, parts[1:])

                # Convert YOLO box to corner coordinates [x1, y1, x2, y2]
                x1 = int((x_center - width / 2) * W)
                y1 = int((y_center - height / 2) * H)
                x2 = int((x_center + width / 2) * W)
                y2 = int((y_center + height / 2) * H)
                input_box = np.array([x1, y1, x2, y2])

                # Predict mask with SAM using the bounding box as a prompt
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )

                # Convert boolean mask to polygon format for YOLOv8
                mask = masks[0]
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Optional: Simplify the polygon. Adjust epsilon for more/less detail.
                    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
                    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    if len(approx_contour) < 3: continue # A polygon needs at least 3 points

                    # Normalize points and format for YOLO segmentation
                    polygon_normalized = []
                    for point in approx_contour.squeeze(1):
                        norm_x = point[0] / W
                        norm_y = point[1] / H
                        polygon_normalized.append(f"{norm_x:.6f}")
                        polygon_normalized.append(f"{norm_y:.6f}")
                    
                    f_out.write(f"{class_id} {' '.join(polygon_normalized)}\n")

        # Progress indicator
        print(f"  ({i+1}/{len(image_files)}) Processed: {image_filename}")

# --- 4. MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    for subset in SUBSETS:
        process_subset(subset)
    print("\nConversion complete!")
    print(f"New segmentation labels are saved in '{OUTPUT_SEG_DIR}' directory.")