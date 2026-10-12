from ultralytics import YOLO


class VehicleDetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame, confidence=0.4):

        results = self.model(
            frame,
            conf=confidence,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence_score = float(box.conf[0])
                class_id = int(box.cls[0])

                detections.append({
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),
                    "confidence": confidence_score,
                    "class_id": class_id
                })

        return detections