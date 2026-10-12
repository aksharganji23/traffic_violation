VIDEO_SOURCE = "input/traffic.mp4"

# Set to 0 for webcam
# VIDEO_SOURCE = 0

VEHICLE_MODEL = "models/vehicle_model.pt"
HELMET_MODEL = "models/helmet_model.pt"
PLATE_MODEL = "models/plate_model.pt"

CONFIDENCE_THRESHOLD = 0.40

OUTPUT_DIR = "output/violations"

# Traffic stop line
STOP_LINE_Y = 350

# Expected traffic direction
# "left_to_right" or "right_to_left"
EXPECTED_DIRECTION = "left_to_right"

# Maximum number of riders allowed
MAX_RIDERS = 2