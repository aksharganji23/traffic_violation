import cv2
import os
from datetime import datetime


def save_evidence(
    frame,
    vehicle_number,
    violation_type,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    evidence = frame.copy()

    height, width = evidence.shape[:2]

    cv2.rectangle(
        evidence,
        (0, 0),
        (width, 100),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        evidence,
        "TRAFFIC VIOLATION",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        evidence,
        f"Violation: {violation_type}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        evidence,
        f"Vehicle: {vehicle_number}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"{violation_type}_"
        f"{vehicle_number}_"
        f"{timestamp}.jpg"
    )

    path = os.path.join(
        output_dir,
        filename
    )

    cv2.imwrite(
        path,
        evidence
    )

    return path