import cv2
import time
import torch
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort

# === Class for Each Person in Queue ===
class Person:
    def __init__(self, object_id, entry_time):
        self.object_id = object_id
        self.entry_time = entry_time
        self.priority = None

    def __repr__(self):
        return f"Person(ID={self.object_id}, Entry={self.entry_time}, Priority={self.priority})"

# === Queue Manager using OOP ===
class QueueManager:
    def __init__(self):
        self.person_dict = {}
        self.queue = deque()

    def add_person(self, object_id):
        if object_id not in self.person_dict:
            new_person = Person(object_id, time.time())
            self.person_dict[object_id] = new_person
            self.queue.append(object_id)
            self._assign_priorities()
            print(f"[INFO] Person {object_id} added.")

    def remove_person(self, object_id):
        if object_id in self.person_dict:
            self.queue.remove(object_id)
            del self.person_dict[object_id]
            self._assign_priorities()
            print(f"[INFO] Person {object_id} removed.")

    def _assign_priorities(self):
        for idx, obj_id in enumerate(self.queue):
            self.person_dict[obj_id].priority = idx + 1

    def print_queue(self):a
        print("\nCurrent Queue:")
        for obj_id in self.queue:
            print(self.person_dict[obj_id])


# === Load YOLOv5 and Deep SORT ===
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.eval()
deep_sort = DeepSort(max_age=30)

cap = cv2.VideoCapture(0)
queue_manager = QueueManager()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    detections = []

    for *xyxy, conf, cls in results.xyxy[0]:
        if int(cls) == 0 and conf > 0.5:  # person class
            x1, y1, x2, y2 = map(int, xyxy)
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], float(conf), 'person'))

    tracks = deep_sort.update_tracks(detections, frame=frame)

    current_ids = set()
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        current_ids.add(track_id)
        queue_manager.add_person(track_id)
        bbox = track.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Remove people who left the frame
    existing_ids = set(queue_manager.person_dict.keys())
    for missing_id in existing_ids - current_ids:
        queue_manager.remove_person(missing_id)

    queue_manager.print_queue()

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

