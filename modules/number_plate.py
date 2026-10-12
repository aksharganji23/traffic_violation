import cv2
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


# ============================================================
# INDIAN VEHICLE REGISTRATION PATTERN
# ============================================================

INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL",
    "DN", "GA", "GJ", "HP", "HR", "JH", "JK", "KA",
    "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ",
    "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR",
    "TS", "UK", "UP", "WB"
}


@dataclass
class PlateResult:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    readable: bool


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_ocr_text(text: str) -> str:

    if not text:
        return ""

    text = text.upper()

    # Remove everything except letters and numbers
    text = re.sub(r"[^A-Z0-9]", "", text)

    # Common OCR mistakes
    replacements = {
        " ": "",
        "|": "1",
        "I": "1",
        "L": "1",
        "O": "0",
        "Q": "0",
        "Z": "2",
        "S": "5",
        "B": "8",
    }

    # Do NOT blindly replace every character.
    # First/letter positions need different handling.

    cleaned = ""

    for ch in text:
        cleaned += ch

    return cleaned


# ============================================================
# INDIAN PLATE FORMAT VALIDATION
# ============================================================

def plate_format_score(text: str) -> float:

    if not text:
        return 0.0

    text = normalize_ocr_text(text)

    if len(text) < 7 or len(text) > 12:
        return 0.0

    score = 0.0

    # --------------------------------------------------------
    # Standard Indian private vehicle format:
    #
    # KA01AB1234
    # TS09EF1234
    # AP28CD5678
    # --------------------------------------------------------

    if len(text) >= 4:

        state = text[:2]

        if state in INDIAN_STATE_CODES:
            score += 0.40

        if text[2:4].isdigit():
            score += 0.20

    # Last portion should contain digits
    digit_count = sum(c.isdigit() for c in text)
    letter_count = sum(c.isalpha() for c in text)

    if digit_count >= 4:
        score += 0.20

    if letter_count >= 3:
        score += 0.10

    # Strong overall pattern
    pattern = re.match(
        r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$",
        text
    )

    if pattern:
        score += 0.30

    return min(score, 1.0)


# ============================================================
# POSITION-AWARE OCR NORMALIZATION
# ============================================================

def correct_plate_by_position(text: str) -> str:

    text = normalize_ocr_text(text)

    if len(text) < 7:
        return text

    chars = list(text)

    # First two positions should normally be letters
    letter_map = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "8": "B",
    }

    for i in range(min(2, len(chars))):
        if chars[i].isdigit() and chars[i] in letter_map:
            chars[i] = letter_map[chars[i]]

    # Positions 2 and 3 normally digits
    digit_map = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
    }

    for i in range(2, min(4, len(chars))):

        if chars[i].isalpha() and chars[i] in digit_map:
            chars[i] = digit_map[chars[i]]

    return "".join(chars)


# ============================================================
# IMAGE QUALITY
# ============================================================

def image_quality_score(crop):

    if crop is None or crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Sharpness
    sharpness = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # Contrast
    contrast = gray.std()

    # Resolution
    h, w = gray.shape

    resolution_score = min(
        1.0,
        (w * h) / (100 * 40)
    )

    sharp_score = min(
        1.0,
        sharpness / 150.0
    )

    contrast_score = min(
        1.0,
        contrast / 60.0
    )

    return (
        0.35 * sharp_score +
        0.35 * contrast_score +
        0.30 * resolution_score
    )


# ============================================================
# UPSCALE
# ============================================================

def upscale_plate(crop, scale=6):

    return cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


# ============================================================
# SHARPEN
# ============================================================

def sharpen(image):

    gaussian = cv2.GaussianBlur(
        image,
        (0, 0),
        3
    )

    return cv2.addWeighted(
        image,
        1.7,
        gaussian,
        -0.7,
        0
    )


# ============================================================
# GENERATE OCR VARIANTS
# ============================================================

def generate_ocr_variants(crop):

    variants = []

    if crop is None or crop.size == 0:
        return variants

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    enlarged = upscale_plate(crop, 6)

    variants.append(
        ("original", enlarged)
    )

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    variants.append(
        ("gray", gray)
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    clahe_img = clahe.apply(gray)

    variants.append(
        ("clahe", clahe_img)
    )

    # --------------------------------------------------------
    # Bilateral
    # --------------------------------------------------------

    bilateral = cv2.bilateralFilter(
        gray,
        9,
        75,
        75
    )

    variants.append(
        ("bilateral", bilateral)
    )

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    sharp = sharpen(clahe_img)

    variants.append(
        ("sharp", sharp)
    )

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        sharp,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(
        ("otsu", otsu)
    )

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )

    variants.append(
        ("adaptive", adaptive)
    )

    # --------------------------------------------------------
    # Blackhat for dark characters
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 5)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    variants.append(
        ("blackhat", blackhat)
    )

    return variants


# ============================================================
# OCR
# ============================================================

def run_ocr(
    ocr_reader,
    image
):

    try:

        results = ocr_reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

    except Exception:
        return []

    output = []

    for result in results:

        if len(result) < 3:
            continue

        text = result[1]

        confidence = float(result[2])

        text = normalize_ocr_text(text)

        if not text:
            continue

        output.append(
            (
                text,
                confidence
            )
        )

    return output


# ============================================================
# OCR ENSEMBLE
# ============================================================

def recognize_plate(
    crop,
    ocr_reader
):

    if crop is None or crop.size == 0:
        return "", 0.0, False

    quality = image_quality_score(crop)

    variants = generate_ocr_variants(crop)

    candidates = []

    for variant_name, variant in variants:

        results = run_ocr(
            ocr_reader,
            variant
        )

        for text, ocr_conf in results:

            corrected = correct_plate_by_position(
                text
            )

            format_score = plate_format_score(
                corrected
            )

            final_score = (
                0.45 * ocr_conf +
                0.35 * format_score +
                0.20 * quality
            )

            candidates.append(
                {
                    "text": corrected,
                    "ocr_conf": ocr_conf,
                    "format_score": format_score,
                    "quality": quality,
                    "score": final_score,
                    "variant": variant_name
                }
            )

    if not candidates:
        return "", 0.0, False

    # --------------------------------------------------------
    # Group similar OCR results
    # --------------------------------------------------------

    groups = {}

    for candidate in candidates:

        text = candidate["text"]

        if len(text) < 7:
            continue

        groups.setdefault(
            text,
            []
        ).append(candidate)

    if not groups:
        return "", 0.0, False

    # --------------------------------------------------------
    # Consensus voting
    # --------------------------------------------------------

    ranked = []

    for text, items in groups.items():

        best = max(
            items,
            key=lambda x: x["score"]
        )

        average_score = sum(
            x["score"] for x in items
        ) / len(items)

        consensus_bonus = min(
            0.20,
            len(items) * 0.04
        )

        total_score = (
            0.65 * best["score"] +
            0.35 * average_score +
            consensus_bonus
        )

        ranked.append(
            (
                text,
                min(total_score, 1.0),
                len(items)
            )
        )

    ranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_text, best_score, votes = ranked[0]

    # --------------------------------------------------------
    # STRICT READABILITY DECISION
    # --------------------------------------------------------
    #
    # Do not guess.
    # --------------------------------------------------------

    readable = (
        best_score >= 0.72
        and plate_format_score(best_text) >= 0.45
        and len(best_text) >= 7
        and votes >= 2
    )

    if not readable:

        return (
            "NOT READABLE",
            best_score,
            False
        )

    return (
        best_text,
        best_score,
        True
    )


# ============================================================
# PLATE CROP
# ============================================================

def crop_bbox(
    image,
    bbox,
    padding=0.08
):

    h, w = image.shape[:2]

    x1, y1, x2, y2 = map(
        int,
        bbox
    )

    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * padding)
    py = int(bh * padding)

    x1 = max(
        0,
        x1 - px
    )

    y1 = max(
        0,
        y1 - py
    )

    x2 = min(
        w,
        x2 + px
    )

    y2 = min(
        h,
        y2 + py
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# MAIN PLATE DETECTION
# ============================================================

def detect_and_read_plates(
    image,
    vehicle_bbox,
    plate_model,
    ocr_reader,
    save_debug=False,
    debug_dir="output/plates"
):

    vehicle_crop = crop_bbox(
        image,
        vehicle_bbox,
        padding=0.05
    )

    if vehicle_crop is None:
        return None

    vh, vw = vehicle_crop.shape[:2]

    # --------------------------------------------------------
    # Multiple scales
    # --------------------------------------------------------

    scales = [
        1.0,
        1.5,
        2.0
    ]

    detections = []

    for scale in scales:

        if scale == 1.0:

            search_image = vehicle_crop

        else:

            search_image = cv2.resize(
                vehicle_crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        try:

            results = plate_model.predict(
                search_image,
                imgsz=640,
                conf=0.15,
                iou=0.45,
                max_det=10,
                verbose=False
            )

        except Exception:
            continue

        if not results:
            continue

        result = results[0]

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Convert scaled coordinates
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            # Convert to original image coordinates
            vx1, vy1, _, _ = map(
                int,
                vehicle_bbox
            )

            x1 += vx1
            x2 += vx1
            y1 += vy1
            y2 += vy1

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                image.shape[1],
                x2
            )

            y2 = min(
                image.shape[0],
                y2
            )

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            detections.append(
                {
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "confidence": confidence,
                    "crop": crop
                }
            )

    if not detections:
        return None

    # --------------------------------------------------------
    # Remove duplicate detections
    # --------------------------------------------------------

    detections.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    best_result = None

    for detection in detections:

        text, score, readable = recognize_plate(
            detection["crop"],
            ocr_reader
        )

        candidate = {
            "bbox": detection["bbox"],
            "det_conf": detection["confidence"],
            "text": text,
            "ocr_score": score,
            "readable": readable
        }

        if best_result is None:

            best_result = candidate

        else:

            # Prefer readable result
            if readable and not best_result["readable"]:

                best_result = candidate

            elif (
                readable == best_result["readable"]
                and score > best_result["ocr_score"]
            ):

                best_result = candidate

    return best_result