from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model(
    source="mug_video_2.mp4",
    save=True,
    conf=0.15
)

print("Video processing completed!")