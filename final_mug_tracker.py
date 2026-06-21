import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.track(
    source="mug_video_2.mp4",
    persist=True,
    conf=0.15,
    classes=[41],  
    stream=True
)

for r in results:
    frame = r.plot()

    cv2.imshow("Coffee Mug Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
