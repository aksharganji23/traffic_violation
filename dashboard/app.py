import streamlit as st
import cv2
import numpy as np
import pandas as pd
import re

from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Traffic Violation Detection",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

VEHICLE_CONF = 0.30
POSE_CONF = 0.30
HELMET_CONF = 0.35
PLATE_CONF = 0.30


# ============================================================
# FAST MODEL LOADERS
# ============================================================
# IMPORTANT:
# Models are NOT loaded when the website starts.
# They load only when Analyze is clicked.
# st.cache_resource keeps them in memory afterwards.
# ============================================================


@st.cache_resource(show_spinner=False)
def get_vehicle_model():

    return YOLO("yolo11n.pt")


@st.cache_resource(show_spinner=False)
def get_pose_model():

    return YOLO("yolo11n-pose.pt")


@st.cache_resource(show_spinner=False)
def get_helmet_model():

    model_path = hf_hub_download(
        repo_id="nnsohamnn/helmet-detection-yolo11",
        filename="yolov11m(100epochs).pt"
    )

    return YOLO(model_path)


@st.cache_resource(show_spinner=False)
def get_plate_model():

    model_path = hf_hub_download(
        repo_id="joker5914/yolov8n-license-plate",
        filename="best.pt"
    )

    return YOLO(model_path)


@st.cache_resource(show_spinner=False)
def get_ocr_reader():

    import easyocr

    return easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False
    )


# ============================================================
# IMAGE HELPERS
# ============================================================


def clamp_box(box, width, height):

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))

    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    return x1, y1, x2, y2


def read_uploaded_image(uploaded_file):

    data = np.asarray(
        bytearray(uploaded_file.getbuffer()),
        dtype=np.uint8
    )

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )


# ============================================================
# VEHICLE DETECTION
# ============================================================


def detect_vehicles(image, model):

    results = model.predict(
        source=image,
        conf=VEHICLE_CONF,
        imgsz=640,
        verbose=False
    )

    vehicles = []

    if not results:
        return vehicles

    result = results[0]

    boxes = getattr(result, "boxes", None)

    if boxes is None:
        return vehicles

    h, w = image.shape[:2]

    class_names = {
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    }

    for box in boxes:

        try:

            class_id = int(
                box.cls[0].item()
            )

            if class_id not in class_names:
                continue

            confidence = float(
                box.conf[0].item()
            )

            coords = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = clamp_box(
                coords,
                w,
                h
            )

            vehicles.append({
                "bbox": (x1, y1, x2, y2),
                "type": class_names[class_id],
                "confidence": confidence
            })

        except Exception:
            continue

    return vehicles


# ============================================================
# POSE / PERSON DETECTION
# ============================================================


def detect_people(image, model):

    results = model.predict(
        source=image,
        conf=POSE_CONF,
        imgsz=640,
        verbose=False
    )

    people = []

    if not results:
        return people

    result = results[0]

    boxes = getattr(
        result,
        "boxes",
        None
    )

    keypoints_object = getattr(
        result,
        "keypoints",
        None
    )

    if boxes is None:
        return people

    h, w = image.shape[:2]

    for index, box in enumerate(boxes):

        try:

            class_id = int(
                box.cls[0].item()
            )

            if class_id != 0:
                continue

            confidence = float(
                box.conf[0].item()
            )

            coords = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = clamp_box(
                coords,
                w,
                h
            )

            keypoints = None

            if keypoints_object is not None:

                try:

                    keypoints = (
                        keypoints_object.xy[index]
                        .cpu()
                        .numpy()
                    )

                except Exception:
                    pass

            people.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "keypoints": keypoints
            })

        except Exception:
            continue

    return people


# ============================================================
# TRIPLE RIDING
# ============================================================


def point_inside(point, box):

    if point is None:
        return False

    x, y = point

    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


def belongs_to_motorcycle(person, motorcycle_box):

    px1, py1, px2, py2 = person["bbox"]

    center = (
        (px1 + px2) / 2,
        (py1 + py2) / 2
    )

    mx1, my1, mx2, my2 = motorcycle_box

    mw = mx2 - mx1
    mh = my2 - my1

    # Expanded region around motorcycle
    expanded = (
        mx1 - int(mw * 0.60),
        my1 - int(mh * 0.60),
        mx2 + int(mw * 0.60),
        my2 + int(mh * 0.40)
    )

    # Person center
    if point_inside(center, expanded):
        return True

    # Pose keypoints
    keypoints = person.get("keypoints")

    if keypoints is not None:

        # Nose, shoulders, hips, ankles
        indices = [
            0,
            5,
            6,
            11,
            12,
            15,
            16
        ]

        for index in indices:

            try:

                x = float(keypoints[index][0])
                y = float(keypoints[index][1])

                if point_inside(
                    (x, y),
                    expanded
                ):
                    return True

            except Exception:
                continue

    return False


def detect_triple_riding(
    motorcycles,
    people
):

    results = []

    for motorcycle in motorcycles:

        riders = []

        for person in people:

            if belongs_to_motorcycle(
                person,
                motorcycle["bbox"]
            ):
                riders.append(person)

        count = len(riders)

        results.append({
            "bbox": motorcycle["bbox"],
            "riders": riders,
            "rider_count": count,
            "violation": count >= 3
        })

    return results


# ============================================================
# HELMET DETECTION
# ============================================================


def detect_helmets(image, model):

    results = model.predict(
        source=image,
        conf=HELMET_CONF,
        imgsz=640,
        verbose=False
    )

    detections = []

    if not results:
        return detections

    result = results[0]

    boxes = getattr(
        result,
        "boxes",
        None
    )

    if boxes is None:
        return detections

    h, w = image.shape[:2]

    for box in boxes:

        try:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            coords = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = clamp_box(
                coords,
                w,
                h
            )

            label = str(
                model.names[class_id]
            )

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "confidence": confidence
            })

        except Exception:
            continue

    return detections




# ============================================================
# SERIOUS LICENSE PLATE PIPELINE
# ============================================================
# Strategy:
# 1. Detect plates globally.
# 2. Re-run the plate detector inside EVERY vehicle ROI.
# 3. Upscale small vehicle ROIs before plate detection.
# 4. Use OpenCV plate-shape fallback when YOLO misses a tiny plate.
# 5. OCR every plausible crop using several preprocessing variants.
# 6. Validate against Indian registration formats.
# 7. Associate each plate with the nearest vehicle.
# 8. Never invent a plate number when evidence is weak.
# ============================================================


INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL",
    "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA",
    "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ",
    "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR",
    "TS", "UK", "UP", "WB"
}


def clean_plate(text):
    text = str(text).upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def normalize_plate_text(text):
    """
    Position-aware correction only.
    We do NOT blindly convert all O->0 / I->1 because that
    can destroy legitimate letters.
    """
    chars = list(clean_plate(text))

    if len(chars) < 7:
        return "".join(chars)

    letter_map = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "8": "B"
    }

    digit_map = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8"
    }

    # First two characters are normally state letters.
    for i in range(min(2, len(chars))):
        if chars[i].isdigit():
            chars[i] = letter_map.get(chars[i], chars[i])

    # Next one/two characters are normally district digits.
    for i in range(2, min(4, len(chars))):
        if chars[i].isalpha():
            chars[i] = digit_map.get(chars[i], chars[i])

    return "".join(chars)


def plate_format_score(text):
    """
    Returns a structural score, not an OCR accuracy claim.
    Supports normal Indian registrations and BH series.
    """
    text = normalize_plate_text(text)

    if len(text) < 7 or len(text) > 12:
        return 0.0

    # Standard:
    # KA01AB1234
    # TS09EF1234
    standard = re.fullmatch(
        r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}",
        text
    )

    # Bharat Series:
    # 22BH1234AA
    bh = re.fullmatch(
        r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}",
        text
    )

    if standard:
        return 1.0

    if bh:
        return 1.0

    score = 0.0

    if text[:2] in INDIAN_STATE_CODES:
        score += 0.45

    if len(text) >= 4 and text[2:4].isdigit():
        score += 0.20

    digit_count = sum(c.isdigit() for c in text)
    letter_count = sum(c.isalpha() for c in text)

    if digit_count >= 4:
        score += 0.20

    if letter_count >= 3:
        score += 0.15

    return min(score, 1.0)


def crop_box_with_padding(image, bbox, padding=0.08):
    h, w = image.shape[:2]

    x1, y1, x2, y2 = map(int, bbox)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    px = int(bw * padding)
    py = int(bh * padding)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def plate_center(bbox):
    x1, y1, x2, y2 = bbox
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def center_distance_normalized(plate_bbox, vehicle_bbox):
    px, py = plate_center(plate_bbox)

    vx1, vy1, vx2, vy2 = vehicle_bbox

    vw = max(1, vx2 - vx1)
    vh = max(1, vy2 - vy1)

    vcx = (vx1 + vx2) / 2.0
    vcy = (vy1 + vy2) / 2.0

    dx = (px - vcx) / vw
    dy = (py - vcy) / vh

    return float(np.sqrt(dx * dx + dy * dy))


def plate_quality(crop):
    if crop is None or crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    contrast = float(gray.std())

    # Tiny crops get penalized.
    pixel_score = min(
        1.0,
        (w * h) / 5000.0
    )

    sharp_score = min(
        1.0,
        sharpness / 180.0
    )

    contrast_score = min(
        1.0,
        contrast / 60.0
    )

    return (
        0.40 * sharp_score +
        0.30 * contrast_score +
        0.30 * pixel_score
    )


# ============================================================
# YOLO PLATE DETECTION
# ============================================================

def _yolo_plate_boxes(search_image, model, conf=0.12):
    detections = []

    try:
        results = model.predict(
            source=search_image,
            conf=conf,
            iou=0.45,
            imgsz=640,
            max_det=20,
            verbose=False
        )
    except Exception:
        return detections

    if not results:
        return detections

    boxes = getattr(results[0], "boxes", None)

    if boxes is None:
        return detections

    h, w = search_image.shape[:2]

    for box in boxes:
        try:
            coords = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = clamp_box(
                coords,
                w,
                h
            )

            confidence = float(
                box.conf[0].item()
            )

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "method": "YOLO"
            })

        except Exception:
            continue

    return detections


def detect_plates_yolo_global(image, model):
    """
    Global pass. Lower confidence is intentional because the
    dedicated plate detector is followed by OCR validation.
    """
    raw = _yolo_plate_boxes(
        image,
        model,
        conf=0.10
    )

    return [
        {
            "bbox": d["bbox"],
            "confidence": d["confidence"],
            "method": d["method"]
        }
        for d in raw
    ]


def detect_plates_inside_vehicle(
    image,
    vehicle,
    model
):
    """
    Critical improvement:
    run the plate detector on an enlarged vehicle crop.
    This is much better for tiny motorcycle plates than
    running the detector only on the full image.
    """
    vehicle_bbox = vehicle["bbox"]

    crop = crop_box_with_padding(
        image,
        vehicle_bbox,
        padding=0.04
    )

    if crop is None:
        return []

    vh, vw = crop.shape[:2]

    if vh < 10 or vw < 10:
        return []

    detections = []

    # Several enlargement factors help very small plates.
    for scale in (1.5, 2.0, 2.5):

        enlarged = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        local = _yolo_plate_boxes(
            enlarged,
            model,
            conf=0.08
        )

        vx1, vy1, vx2, vy2 = vehicle_bbox

        for d in local:

            lx1, ly1, lx2, ly2 = d["bbox"]

            # Undo enlargement.
            lx1 = int(lx1 / scale)
            ly1 = int(ly1 / scale)
            lx2 = int(lx2 / scale)
            ly2 = int(ly2 / scale)

            # crop_box_with_padding adds a small offset.
            pad_x = int((vx2 - vx1) * 0.04)
            pad_y = int((vy2 - vy1) * 0.04)

            gx1 = vx1 - pad_x + lx1
            gy1 = vy1 - pad_y + ly1
            gx2 = vx1 - pad_x + lx2
            gy2 = vy1 - pad_y + ly2

            gx1, gy1, gx2, gy2 = clamp_box(
                (gx1, gy1, gx2, gy2),
                image.shape[1],
                image.shape[0]
            )

            if gx2 <= gx1 or gy2 <= gy1:
                continue

            detections.append({
                "bbox": (gx1, gy1, gx2, gy2),
                "confidence": d["confidence"],
                "method": "YOLO-VEHICLE-ROI",
                "vehicle_index": vehicle.get(
                    "index",
                    -1
                )
            })

    return detections


# ============================================================
# OPENCV FALLBACK FOR TINY PLATES
# ============================================================

def detect_plate_candidates_opencv(
    image,
    vehicle,
    max_candidates=12
):
    """
    Fallback only.
    It proposes plate-shaped regions; OCR + format validation
    decides whether any proposal is actually a plate.
    """

    vehicle_bbox = vehicle["bbox"]

    crop = crop_box_with_padding(
        image,
        vehicle_bbox,
        padding=0.03
    )

    if crop is None or crop.size == 0:
        return []

    original_h, original_w = crop.shape[:2]

    scale = 3.0

    work = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        work,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.bilateralFilter(
        gray,
        7,
        50,
        50
    )

    variants = []

    # Dark-character-on-bright-plate detector.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (21, 7)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, blackhat_thr = cv2.threshold(
        blackhat,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(blackhat_thr)

    # Bright plate regions.
    _, bright = cv2.threshold(
        gray,
        150,
        255,
        cv2.THRESH_BINARY
    )

    variants.append(bright)

    # Edge-based fallback.
    edges = cv2.Canny(
        gray,
        80,
        180
    )

    variants.append(edges)

    candidates = []

    for mask in variants:

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (17, 5)
            ),
            iterations=1
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w <= 0 or h <= 0:
                continue

            area = w * h
            image_area = work.shape[0] * work.shape[1]

            area_ratio = area / max(
                1,
                image_area
            )

            aspect = w / max(
                1.0,
                float(h)
            )

            # Motorcycle plates can be somewhat wider,
            # while perspective can make them less wide.
            if not (
                1.5 <= aspect <= 7.5
            ):
                continue

            if not (
                0.0004 <= area_ratio <= 0.15
            ):
                continue

            # Plates are normally in the lower/middle vehicle
            # area, not at the extreme top.
            center_y = y + h / 2

            if center_y < work.shape[0] * 0.35:
                continue

            rect_area = float(w * h)

            contour_area = cv2.contourArea(
                contour
            )

            rectangularity = (
                contour_area / rect_area
                if rect_area > 0
                else 0
            )

            if rectangularity < 0.15:
                continue

            score = (
                min(aspect / 4.0, 1.0) * 0.30
                +
                min(area_ratio / 0.03, 1.0) * 0.25
                +
                min(rectangularity / 0.70, 1.0) * 0.25
                +
                min(
                    center_y / work.shape[0],
                    1.0
                ) * 0.20
            )

            # Back to crop coordinates.
            cx1 = int(x / scale)
            cy1 = int(y / scale)
            cx2 = int((x + w) / scale)
            cy2 = int((y + h) / scale)

            # Add a little padding.
            pad_x = max(
                2,
                int((cx2 - cx1) * 0.10)
            )
            pad_y = max(
                2,
                int((cy2 - cy1) * 0.15)
            )

            cx1 = max(0, cx1 - pad_x)
            cy1 = max(0, cy1 - pad_y)
            cx2 = min(original_w, cx2 + pad_x)
            cy2 = min(original_h, cy2 + pad_y)

            vx1, vy1, vx2, vy2 = vehicle_bbox

            vehicle_pad_x = int(
                (vx2 - vx1) * 0.03
            )

            vehicle_pad_y = int(
                (vy2 - vy1) * 0.03
            )

            gx1 = vx1 - vehicle_pad_x + cx1
            gy1 = vy1 - vehicle_pad_y + cy1
            gx2 = vx1 - vehicle_pad_x + cx2
            gy2 = vy1 - vehicle_pad_y + cy2

            gx1, gy1, gx2, gy2 = clamp_box(
                (gx1, gy1, gx2, gy2),
                image.shape[1],
                image.shape[0]
            )

            if gx2 <= gx1 or gy2 <= gy1:
                continue

            candidates.append({
                "bbox": (
                    gx1,
                    gy1,
                    gx2,
                    gy2
                ),
                "confidence": float(
                    0.20 + 0.45 * score
                ),
                "method": "OPENCV-FALLBACK",
                "vehicle_index": vehicle.get(
                    "index",
                    -1
                )
            })

    # Highest scoring first.
    candidates.sort(
        key=lambda d: d["confidence"],
        reverse=True
    )

    # Simple NMS.
    kept = []

    for candidate in candidates:

        duplicate = False

        for existing in kept:

            if bbox_iou(
                candidate["bbox"],
                existing["bbox"]
            ) > 0.45:

                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

        if len(kept) >= max_candidates:
            break

    return kept


# ============================================================
# OCR PREPROCESSING
# ============================================================

def sharpen_image(image):
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        2.0
    )

    return cv2.addWeighted(
        image,
        1.8,
        blurred,
        -0.8,
        0
    )


def generate_plate_variants(crop):
    if crop is None or crop.size == 0:
        return []

    # Avoid absurdly large OCR images.
    h, w = crop.shape[:2]

    target_width = max(
        320,
        min(1200, w * 8)
    )

    scale = target_width / max(
        1,
        w
    )

    scale = max(
        3.0,
        min(8.0, scale)
    )

    enlarged = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    bilateral = cv2.bilateralFilter(
        enhanced,
        7,
        60,
        60
    )

    sharp = sharpen_image(
        bilateral
    )

    _, otsu = cv2.threshold(
        sharp,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )

    # Both normal and inverted versions are useful because
    # plate colors/lighting vary.
    inverted_otsu = cv2.bitwise_not(
        otsu
    )

    return [
        ("color", enlarged),
        ("gray", gray),
        ("clahe", enhanced),
        ("sharp", sharp),
        ("otsu", otsu),
        ("adaptive", adaptive),
        ("otsu_inverted", inverted_otsu)
    ]


def edit_distance(a, b):
    """
    Small Levenshtein distance helper used for OCR consensus.
    """
    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):

        current = [i]

        for j, cb in enumerate(b, 1):

            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (
                0 if ca == cb else 1
            )

            current.append(
                min(insert, delete, replace)
            )

        previous = current

    return previous[-1]


def similar_text(a, b):
    if not a or not b:
        return False

    distance = edit_distance(
        a,
        b
    )

    max_len = max(
        len(a),
        len(b)
    )

    return (
        distance <= 1
        if max_len <= 9
        else distance <= 2
    )


def read_plate_crop(
    crop,
    ocr_reader
):
    """
    OCR ensemble.

    Returns:
        text, confidence, format_score, votes
    """

    if crop is None or crop.size == 0:
        return (
            "NOT READABLE",
            0.0,
            0.0,
            0
        )

    quality = plate_quality(
        crop
    )

    variants = generate_plate_variants(
        crop
    )

    raw_candidates = []

    for variant_name, img in variants:

        try:
            results = ocr_reader.readtext(
                img,
                detail=1,
                paragraph=False,
                allowlist=(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789"
                ),
                text_threshold=0.45,
                low_text=0.25,
                link_threshold=0.20
            )
        except Exception:
            continue

        for item in results:

            if not isinstance(
                item,
                (list, tuple)
            ):
                continue

            if len(item) < 3:
                continue

            try:
                text = normalize_plate_text(
                    item[1]
                )

                ocr_conf = float(
                    item[2]
                )

            except Exception:
                continue

            if len(text) < 7:
                continue

            if len(text) > 12:
                continue

            format_score = plate_format_score(
                text
            )

            # We retain partially valid candidates for
            # consensus rather than rejecting them immediately.
            score = (
                0.50 * ocr_conf
                +
                0.30 * format_score
                +
                0.20 * quality
            )

            raw_candidates.append({
                "text": text,
                "ocr": ocr_conf,
                "format": format_score,
                "score": score,
                "variant": variant_name
            })

    if not raw_candidates:
        return (
            "NOT READABLE",
            0.0,
            0.0,
            0
        )

    # --------------------------------------------------------
    # Cluster near-identical OCR readings.
    # Example:
    # KA01AB1234
    # KA01AB1234
    # KA01AB1234
    # --------------------------------------------------------

    clusters = []

    for candidate in raw_candidates:

        placed = False

        for cluster in clusters:

            if similar_text(
                candidate["text"],
                cluster[0]["text"]
            ):
                cluster.append(candidate)
                placed = True
                break

        if not placed:
            clusters.append(
                [candidate]
            )

    ranked_clusters = []

    for cluster in clusters:

        best = max(
            cluster,
            key=lambda x: x["score"]
        )

        avg_score = sum(
            c["score"]
            for c in cluster
        ) / len(cluster)

        votes = len(cluster)

        consensus_bonus = min(
            0.20,
            0.05 * votes
        )

        final_score = min(
            1.0,
            0.60 * best["score"]
            +
            0.40 * avg_score
            +
            consensus_bonus
        )

        ranked_clusters.append({
            "text": best["text"],
            "score": final_score,
            "ocr": best["ocr"],
            "format": best["format"],
            "votes": votes
        })

    ranked_clusters.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = ranked_clusters[0]

    # --------------------------------------------------------
    # STRICT acceptance.
    # The system must prefer "NOT READABLE" to a false plate.
    # --------------------------------------------------------

    readable = (
        best["format"] >= 0.55
        and best["ocr"] >= 0.60
        and best["score"] >= 0.62
        and (
            best["votes"] >= 2
            or (
                best["ocr"] >= 0.82
                and best["format"] >= 0.90
                and best["score"] >= 0.78
            )
        )
    )

    if not readable:
        return (
            "NOT READABLE",
            float(best["score"]),
            float(best["format"]),
            int(best["votes"])
        )

    return (
        best["text"],
        float(best["score"]),
        float(best["format"]),
        int(best["votes"])
    )


# ============================================================
# FINAL PLATE DETECTION + OCR
# ============================================================

def detect_plates(
    image,
    vehicles,
    model,
    ocr_reader
):
    """
    Complete plate pipeline.

    Global YOLO + per-vehicle YOLO + OpenCV fallback.
    Then OCR and vehicle association.
    """

    candidates = []

    # --------------------------------------------------------
    # 1. Global YOLO
    # --------------------------------------------------------

    global_detections = detect_plates_yolo_global(
        image,
        model
    )

    for d in global_detections:
        candidates.append(d)

    # --------------------------------------------------------
    # 2. Vehicle-specific YOLO
    # --------------------------------------------------------

    for vehicle_index, vehicle in enumerate(
        vehicles
    ):

        vehicle["index"] = vehicle_index

        local = detect_plates_inside_vehicle(
            image,
            vehicle,
            model
        )

        for d in local:
            d["vehicle_index"] = vehicle_index
            candidates.append(d)

        # ----------------------------------------------------
        # 3. OpenCV fallback
        # ----------------------------------------------------

        fallback = detect_plate_candidates_opencv(
            image,
            vehicle
        )

        for d in fallback:
            d["vehicle_index"] = vehicle_index
            candidates.append(d)

    if not candidates:
        return []

    # --------------------------------------------------------
    # Deduplicate candidates.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    unique = []

    for candidate in candidates:

        duplicate = False

        for existing in unique:

            if bbox_iou(
                candidate["bbox"],
                existing["bbox"]
            ) > 0.45:

                # Keep the stronger detector result.
                duplicate = True
                break

        if not duplicate:
            unique.append(candidate)

    # --------------------------------------------------------
    # Assign each candidate to a vehicle.
    # --------------------------------------------------------

    for candidate in unique:

        best_vehicle = None
        best_vehicle_score = -999.0

        for index, vehicle in enumerate(
            vehicles
        ):

            vb = vehicle["bbox"]
            pb = candidate["bbox"]

            iou = bbox_iou(
                pb,
                vb
            )

            distance = center_distance_normalized(
                pb,
                vb
            )

            # Strongly prefer plates inside a vehicle.
            score = (
                3.0 * iou
                -
                0.50 * distance
            )

            # If candidate came from this vehicle ROI,
            # give it an association bonus.
            if candidate.get(
                "vehicle_index",
                -1
            ) == index:

                score += 1.0

            if score > best_vehicle_score:
                best_vehicle_score = score
                best_vehicle = index

        candidate["vehicle_index"] = (
            best_vehicle
            if best_vehicle is not None
            else -1
        )

    # --------------------------------------------------------
    # OCR.
    # --------------------------------------------------------

    final_plates = []

    for candidate in unique:

        x1, y1, x2, y2 = candidate[
            "bbox"
        ]

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue

        text, ocr_score, format_score, votes = (
            read_plate_crop(
                crop,
                ocr_reader
            )
        )

        candidate["text"] = text
        candidate["ocr_score"] = ocr_score
        candidate["format_score"] = format_score
        candidate["votes"] = votes
        candidate["quality"] = plate_quality(
            crop
        )
        candidate["crop"] = crop

        # Keep YOLO detections and good fallback candidates.
        # Very weak fallback candidates with no OCR evidence
        # are removed so the UI doesn't fill with noise.
        if (
            candidate["method"] == "OPENCV-FALLBACK"
            and text == "NOT READABLE"
            and ocr_score < 0.30
        ):
            continue

        final_plates.append(
            candidate
        )

    # --------------------------------------------------------
    # Keep the strongest plate per vehicle.
    # --------------------------------------------------------

    best_by_vehicle = {}

    for plate in final_plates:

        vehicle_index = plate.get(
            "vehicle_index",
            -1
        )

        if vehicle_index < 0:
            continue

        current = best_by_vehicle.get(
            vehicle_index
        )

        # Readable result always beats unreadable.
        readable = (
            plate["text"] != "NOT READABLE"
        )

        if current is None:
            best_by_vehicle[
                vehicle_index
            ] = plate
            continue

        current_readable = (
            current["text"] != "NOT READABLE"
        )

        if readable and not current_readable:
            best_by_vehicle[
                vehicle_index
            ] = plate

        elif readable == current_readable:

            current_score = (
                current["ocr_score"]
                +
                0.30 * current["confidence"]
            )

            new_score = (
                plate["ocr_score"]
                +
                0.30 * plate["confidence"]
            )

            if new_score > current_score:
                best_by_vehicle[
                    vehicle_index
                ] = plate

    # --------------------------------------------------------
    # Final output ordered by vehicle.
    # --------------------------------------------------------

    final = []

    for vehicle_index in range(
        len(vehicles)
    ):

        plate = best_by_vehicle.get(
            vehicle_index
        )

        if plate is not None:
            final.append(plate)

    return final
# ============================================================
# DRAWING
# ============================================================


def draw_vehicles(
    image,
    vehicles
):

    output = image.copy()

    for vehicle in vehicles:

        x1, y1, x2, y2 = vehicle["bbox"]

        label = (
            f'{vehicle["type"]} '
            f'{vehicle["confidence"]:.2f}'
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    return output


def draw_triple_riding(
    image,
    results
):

    output = image.copy()

    for item in results:

        x1, y1, x2, y2 = item["bbox"]

        count = item["rider_count"]

        if item["violation"]:

            color = (0, 0, 255)

            label = (
                f"TRIPLE RIDING "
                f"({count})"
            )

        else:

            color = (0, 255, 0)

            label = (
                f"{count} RIDER(S)"
            )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    return output


def draw_helmets(
    image,
    detections
):

    output = image.copy()

    for item in detections:

        x1, y1, x2, y2 = item["bbox"]

        label = item["label"]

        confidence = item[
            "confidence"
        ]

        if "without" in label.lower():

            color = (0, 0, 255)

            text = (
                f"NO HELMET "
                f"{confidence:.2f}"
            )

        else:

            color = (0, 255, 0)

            text = (
                f"HELMET "
                f"{confidence:.2f}"
            )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            output,
            text,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return output


def draw_plates(
    image,
    plates
):
    output = image.copy()

    for item in plates:

        x1, y1, x2, y2 = item["bbox"]

        det_conf = item.get(
            "confidence",
            0.0
        )

        text = item.get(
            "text",
            "NOT READABLE"
        )

        if text != "NOT READABLE":
            color = (0, 255, 0)
            label = (
                f"PLATE: {text} "
                f"{item.get('ocr_score', 0.0):.0%}"
            )
        else:
            color = (0, 165, 255)
            label = (
                f"PLATE DETECTED "
                f"{det_conf:.0%} | OCR LOW"
            )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2
        )

    return output


# ============================================================
# COMPLETE IMAGE ANALYSIS
# ============================================================


def analyze_image(image):

    # --------------------------------------------------------
    # Lightweight detection
    # --------------------------------------------------------

    vehicle_model = get_vehicle_model()
    pose_model = get_pose_model()

    vehicles = detect_vehicles(
        image,
        vehicle_model
    )

    people = detect_people(
        image,
        pose_model
    )

    motorcycles = [
        vehicle
        for vehicle in vehicles
        if vehicle["type"] == "Motorcycle"
    ]

    motorcycle_results = (
        detect_triple_riding(
            motorcycles,
            people
        )
    )

    # --------------------------------------------------------
    # Helmet
    # --------------------------------------------------------

    helmet_model = get_helmet_model()

    helmet_detections = detect_helmets(
        image,
        helmet_model
    )

    # --------------------------------------------------------
    # Number plate
    # --------------------------------------------------------

    plate_model = get_plate_model()

    # OCR reader is loaded once and cached.
    ocr_reader = get_ocr_reader()

    plates = detect_plates(
        image,
        vehicles,
        plate_model,
        ocr_reader
    )

    plate_numbers = [
        plate["text"]
        for plate in plates
        if plate.get("text")
        and plate["text"] != "NOT READABLE"
    ]

    # --------------------------------------------------------
    # Draw
    # --------------------------------------------------------

    output = image.copy()

    output = draw_vehicles(
        output,
        vehicles
    )

    output = draw_triple_riding(
        output,
        motorcycle_results
    )

    output = draw_helmets(
        output,
        helmet_detections
    )

    output = draw_plates(
        output,
        plates
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    triple_count = sum(
        1
        for item in motorcycle_results
        if item["violation"]
    )

    no_helmet = sum(
        1
        for item in helmet_detections
        if "without"
        in item["label"].lower()
    )

    helmet = sum(
        1
        for item in helmet_detections
        if "with"
        in item["label"].lower()
        and
        "without"
        not in item["label"].lower()
    )

    return {
        "image": output,
        "vehicles": vehicles,
        "people": people,
        "motorcycles": motorcycles,
        "motorcycle_results": motorcycle_results,
        "helmet_detections": helmet_detections,
        "plates": plates,
        "plate_numbers": plate_numbers,
        "triple_count": triple_count,
        "no_helmet": no_helmet,
        "helmet": helmet
    }


# ============================================================
# PAGE
# ============================================================
# ============================================================
# PAGE
# ============================================================


st.title(
    "🚦 AI Traffic Violation Detection System"
)

st.caption(
    "AI-powered detection of helmet violations, "
    "triple riding and vehicle number plates."
)


# ============================================================
# UPLOAD
# ============================================================


uploaded_file = st.file_uploader(
    "📁 Upload Traffic Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# IMAGE
# ============================================================


if uploaded_file:

    image = read_uploaded_image(
        uploaded_file
    )

    if image is None:

        st.error(
            "❌ Unable to read image."
        )

    else:

        st.subheader(
            "🖼️ Uploaded Image"
        )

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            width="stretch"
        )

        if st.button(
            "🔍 Analyze Traffic Violations",
            type="primary"
        ):

            progress = st.progress(
                0
            )

            status = st.empty()

            status.info(
                "Loading vehicle and pose models..."
            )

            progress.progress(15)

            result = analyze_image(
                image
            )

            progress.progress(100)

            status.success(
                "✅ Analysis completed!"
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.subheader(
                "🎯 AI Detection Result"
            )

            st.image(
                cv2.cvtColor(
                    result["image"],
                    cv2.COLOR_BGR2RGB
                ),
                width="stretch"
            )

            # ------------------------------------------------
            # VIOLATION REPORT
            # ------------------------------------------------

            st.subheader(
                "🚨 Traffic Violation Report"
            )

            violations = 0

            if result["triple_count"] > 0:

                violations += (
                    result["triple_count"]
                )

                st.error(
                    "❌ Triple Riding Detected"
                )

                for item in result[
                    "motorcycle_results"
                ]:

                    if item["violation"]:

                        st.write(
                            f"👥 Riders: "
                            f"{item['rider_count']}"
                        )

            if result["no_helmet"] > 0:

                violations += (
                    result["no_helmet"]
                )

                st.error(
                    "❌ No Helmet Detected"
                )

                st.write(
                    f"🪖 Riders without helmet: "
                    f"{result['no_helmet']}"
                )

            if violations == 0:

                st.success(
                    "✅ No confirmed violation detected."
                )

            # ------------------------------------------------
            # PLATES
            # ------------------------------------------------

            st.subheader(
                "🔢 Vehicle Number Plates"
            )

            if result["plates"]:

                for i, plate in enumerate(
                    result["plates"],
                    start=1
                ):

                    text = plate.get(
                        "text",
                        "NOT READABLE"
                    )

                    vehicle_index = plate.get(
                        "vehicle_index",
                        -1
                    )

                    vehicle_name = (
                        result["vehicles"][vehicle_index]["type"]
                        if 0 <= vehicle_index < len(result["vehicles"])
                        else "Vehicle"
                    )

                    if text != "NOT READABLE":

                        st.success(
                            f"🚗 {vehicle_name} — "
                            f"Plate {i}: **{text}**"
                        )

                        st.caption(
                            f"Detection: {plate['confidence']:.0%} | "
                            f"OCR confidence: {plate['ocr_score']:.0%} | "
                            f"OCR votes: {plate['votes']}"
                        )

                    else:

                        st.warning(
                            f"🚗 {vehicle_name} — "
                            f"Plate {i}: **NOT READABLE**"
                        )

                        st.caption(
                            f"Plate localized ({plate['method']}) but "
                            f"OCR evidence was insufficient. "
                            f"Detection confidence: "
                            f"{plate['confidence']:.0%}"
                        )

                    # Show the actual crop used for OCR.
                    with st.expander(
                        f"🔎 Inspect plate crop {i}"
                    ):

                        crop_rgb = cv2.cvtColor(
                            plate["crop"],
                            cv2.COLOR_BGR2RGB
                        )

                        st.image(
                            crop_rgb,
                            width="stretch"
                        )

            else:

                st.warning(
                    "No reliable license-plate region was found."
                )

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.subheader(
                "📊 Detection Summary"
            )

            c1, c2, c3, c4, c5 = (
                st.columns(5)
            )

            c1.metric(
                "🏍️ Motorcycles",
                len(result["motorcycles"])
            )

            c2.metric(
                "👥 People",
                len(result["people"])
            )

            c3.metric(
                "🪖 Helmet",
                result["helmet"]
            )

            c4.metric(
                "❌ No Helmet",
                result["no_helmet"]
            )

            c5.metric(
                "🚨 Violations",
                violations
            )

            # ------------------------------------------------
            # VEHICLE TABLE
            # ------------------------------------------------

            st.subheader(
                "🚘 Vehicle Analysis"
            )

            rows = []

            for motorcycle_index, item in enumerate(
                result["motorcycle_results"]
            ):

                # Find the corresponding original vehicle index.
                motorcycle_bbox = item["bbox"]

                vehicle_index = -1

                for idx, vehicle in enumerate(
                    result["vehicles"]
                ):
                    if (
                        vehicle["type"] == "Motorcycle"
                        and vehicle["bbox"] == motorcycle_bbox
                    ):
                        vehicle_index = idx
                        break

                associated_plate = "Not Readable"

                for plate in result["plates"]:
                    if plate.get("vehicle_index") == vehicle_index:
                        associated_plate = plate.get(
                            "text",
                            "Not Readable"
                        )
                        break

                rows.append({
                    "Vehicle": "Motorcycle",
                    "Riders": item["rider_count"],
                    "Violation": (
                        "Triple Riding"
                        if item["violation"]
                        else "None"
                    ),
                    "Number Plate": associated_plate
                })

            for vehicle in result[
                "vehicles"
            ]:

                if vehicle["type"] != "Motorcycle":

                    rows.append({
                        "Vehicle": vehicle["type"],
                        "Riders": "-",
                        "Violation": "None",
                        "Number Plate": "-"
                    })

            if rows:

                # Streamlit/PyArrow requires consistent column
                # types. Keep the dashboard display columns as text.
                table_df = pd.DataFrame(rows)

                for column in table_df.columns:
                    table_df[column] = (
                        table_df[column]
                        .fillna("-")
                        .astype(str)
                    )

                st.dataframe(
                    table_df,
                    width="stretch",
                    hide_index=True
                )

            # ------------------------------------------------
            # TECHNICAL DETAILS
            # ------------------------------------------------

            with st.expander(
                "🔧 Technical Details"
            ):

                st.write(
                    f"Vehicles: "
                    f"{len(result['vehicles'])}"
                )

                st.write(
                    f"People: "
                    f"{len(result['people'])}"
                )

                st.write(
                    f"Motorcycles: "
                    f"{len(result['motorcycles'])}"
                )

                st.write(
                    f"Helmet detections: "
                    f"{len(result['helmet_detections'])}"
                )

                st.write(
                    f"License plates: "
                    f"{len(result['plates'])}"
                )
