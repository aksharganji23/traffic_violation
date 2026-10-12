from ultralytics import YOLO


class HelmetDetector:

    def __init__(self, model_path):

        self.model = YOLO(model_path)

    def detect(self, frame):

        results = self.model(
            frame,
            conf=0.4,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = self.model.names[class_id]

                detections.append({
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),
                    "class": class_name,
                    "confidence": confidence
                })

        return detections
def check_no_helmet(detections):

    violations = []

    for detection in detections:

        if detection["class"].lower() in [
            "no_helmet",
            "no helmet",
            "without helmet"
        ]:

            violations.append(detection)

    return violations