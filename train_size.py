from ultralytics import YOLO

def main():
    # 1. Load a pretrained YOLOv8 Large Segmentation model
    model = YOLO("yolov8l-seg.pt")  # segmentation model

    # 2. Train on your custom segmentation dataset
    results = model.train(
        data="data-size.yaml",       # Your custom dataset YAML
        epochs=50,
        imgsz=640,
        batch=16,
        name="steel_segmentation_large_size",
        patience=10,
        device=0                # 0 for GPU, 'cpu' for CPU
    )

    # 3. Validate the model
    metrics = model.val()

    
    pred_results = model.predict(
        source="C:/Users/STUDENT/Downloads/metal/valid/images",  # or test set path
        conf=0.25,
        save=True,
        name="val_predictions_seg_size"
    )

    # 5. Print metrics
    print("Validation Metrics:")
    print(metrics)

if __name__ == "__main__":
    main()
