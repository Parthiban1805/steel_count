from ultralytics import YOLO

def main():
    # 1. Load a pretrained RT-DETR large model
    # The only change is here: "yolov8l.pt" becomes "rtdetr-l.pt"
    # You can also use 'rtdetr-x.pt' for the extra-large model.
    model = YOLO("rtdetr-l.pt") 

    # 2. Train on your custom dataset (no changes needed here)
    results = model.train(
        data="data-size.yaml",       # This YAML file points to your dataset
        epochs=50,
        imgsz=640,
        batch=16,
        name="steel_detector_rtdetr_l_size", # Updated name for clarity
        patience=10,
        device=0                # Use 0 for GPU, 'cpu' for CPU
    )

    # 3. Validate the model (no changes needed)
    # The 'val()' method will automatically use the best weights from training
    metrics = model.val()

    # 4. Predict on validation set and save results (no changes needed)
    pred_results = model.predict(
        source="C:/Users/STUDENT/Downloads/metal/valid/images",
        conf=0.25,
        save=True,
        name="val_predictions_rtdetr_l_size" # Updated name for clarity
    )

    # 5. Print metrics
    print("Validation Metrics:")
    print(f"  Precision-Recall Curve (mAP50-95): {metrics.box.map}")
    print(f"  Mean Average Precision (mAP50): {metrics.box.map50}")
    print(f"  Mean Average Precision (mAP75): {metrics.box.map75}")

if __name__ == "__main__":
    main()