"""Train a YOLOv8 model on your labelled object images.

Example:
    python train_custom_model.py --data custom_dataset.yaml --epochs 100
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train a custom YOLOv8 object detector")
    parser.add_argument("--data", default="custom_dataset.yaml", help="YOLO dataset YAML file")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 chooses a safe batch size")
    args = parser.parse_args()

    dataset_file = Path(args.data)
    if not dataset_file.is_file():
        raise SystemExit(f"Dataset configuration not found: {dataset_file}")

    model = YOLO(args.model)
    model.train(data=str(dataset_file), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=20)
    print("Training complete. Select runs/detect/train/weights/best.pt as the detector model.")


if __name__ == "__main__":
    main()
