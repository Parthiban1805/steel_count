from ultralytics import YOLO

def main():
    # 1. Load a pretrained YOLOv8 model
    model = YOLO("yolov8n-seg.pt")  # not yolov8n.pt
 # You can replace with yolov8s.pt, yolov8m.pt, etc.

    # 2. Train on your custom dataset
    results = model.train(
        data="data.yaml",       # Make sure this path is correct
        epochs=50,
        imgsz=640,
        batch=16,
        name="steel_detector_seg",
        patience=10,
        device=0                # Change to 'cpu' if no GPU
    )

    # 3. Validate the model
    metrics = model.val()

    # 4. Predict on validation set and save results
    pred_results = model.predict(
        source="yolo_data/images/val",
        conf=0.25,
        save=True,
        name="val_predictions"
    )

    # 5. Print metrics
    print("Validation Metrics:")
    print(metrics)

if __name__ == "__main__":
    main()
