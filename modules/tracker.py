import math


class CentroidTracker:

    def __init__(self, max_distance=60):

        self.next_id = 1
        self.objects = {}
        self.max_distance = max_distance

    def centroid(self, bbox):

        x1, y1, x2, y2 = bbox

        return (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )

    def update(self, detections):

        new_objects = {}

        for detection in detections:

            bbox = detection["bbox"]

            cx, cy = self.centroid(bbox)

            matched_id = None
            min_distance = float("inf")

            for object_id, data in self.objects.items():

                old_cx, old_cy = data["centroid"]

                distance = math.sqrt(
                    (cx - old_cx) ** 2 +
                    (cy - old_cy) ** 2
                )

                if distance < min_distance:
                    min_distance = distance
                    matched_id = object_id

            if (
                matched_id is not None
                and min_distance < self.max_distance
            ):

                object_id = matched_id

            else:

                object_id = self.next_id
                self.next_id += 1

            new_objects[object_id] = {
                "bbox": bbox,
                "centroid": (cx, cy),
                "class_id": detection["class_id"],
                "confidence": detection["confidence"]
            }

        self.objects = new_objects

        return self.objects