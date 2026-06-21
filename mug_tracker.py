from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.track(
    source="mug_video_2.mp4",
    save=True,
    persist=True,
    conf=0.15
)

print("Tracking completed!")