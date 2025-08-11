"""
Complete Configuration file for Flask-AI + Security Scanner + OpenAI Integration
PRODUCTION MODE: Stable thresholds for reliable performance
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===========================
# BASE PATHS
# ===========================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# ===========================
# FLASK-AI CONFIGURATION
# ===========================

# Model paths
ANOMALIB_MODEL_PATH = MODELS_DIR / "patchcore.pt"
HRNET_MODEL_PATH = MODELS_DIR / "defect_segmentation_model.pth"

# Detection thresholds - PRODUCTION VALUES
ANOMALY_THRESHOLD = 0.3
DEFECT_CONFIDENCE_THRESHOLD = 0.50

# Device configuration
DEVICE = 'cuda'  # or 'cpu'

# Defect detection classes
SPECIFIC_DEFECT_CLASSES = {
    0: "background",
    1: "damaged",
    2: "missing_component", 
    3: "open",
    4: "scratch",
    5: "stained"
}

DEFECT_COLORS = {
    0: (64, 64, 64),     # Dark gray - background
    1: (255, 0, 0),      # Red - damaged
    2: (255, 255, 0),    # Yellow - missing component
    3: (255, 0, 255),    # Magenta - open
    4: (0, 255, 255),    # Cyan - scratch
    5: (128, 0, 128),    # Purple - stained
}

# Image preprocessing settings
IMAGE_SIZE = (512, 512)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Minimum detection thresholds - PRODUCTION VALUES
MIN_DEFECT_PIXELS = 50
MIN_DEFECT_PERCENTAGE = 0.005  # 0.5%
MIN_BBOX_AREA = 100

# ===========================
# OPENAI CONFIGURATION
# ===========================
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = "gpt-4-turbo"
OPENAI_MAX_TOKENS = 1000
OPENAI_TEMPERATURE = 0.1

# ===========================
# SECURITY SCANNER CONFIGURATION
# ===========================

# Security directories
SECURITY_DIR = BASE_DIR / "security_data"
YARA_RULES_DIR = SECURITY_DIR / "yara_rules"
MALWARE_HASH_FILE = SECURITY_DIR / "malware_hashes.txt"

# File validation settings
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MIN_FILE_SIZE = 100  # 100 bytes
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif'}

# Risk levels
RISK_LEVELS = {
    'CLEAN': 0,
    'LOW': 1, 
    'MEDIUM': 2,
    'HIGH': 3,
    'CRITICAL': 4
}

# Error codes
ERROR_CODES = {
    'FILE_TOO_LARGE': 'E001',
    'INVALID_FILE_TYPE': 'E002', 
    'SCAN_TIMEOUT': 'E003',
    'YARA_ERROR': 'E004',
    'INTERNAL_ERROR': 'E999'
}

# Hash algorithms
HASH_ALGORITHMS = ['md5', 'sha1', 'sha256']

# YARA rules files
YARA_RULES_FILES = {
    'light_scan': 'light_scan_rules.yar',
    'full_scan': 'full_scan_rules.yar'
}

# MIME type mapping
MIME_TYPE_MAPPING = {
    '.jpg': ['image/jpeg'],
    '.jpeg': ['image/jpeg'],
    '.png': ['image/png'],
    '.gif': ['image/gif'],
    '.bmp': ['image/bmp', 'image/x-ms-bmp'],
    '.tiff': ['image/tiff'],
    '.tif': ['image/tiff']
}

# Magic byte signatures
MAGIC_SIGNATURES = {
    'JPEG': [b'\xFF\xD8\xFF'],
    'PNG': [b'\x89PNG\r\n\x1a\n'],
    'GIF87a': [b'GIF87a'],
    'GIF89a': [b'GIF89a'],
    'BMP': [b'BM'],
    'TIFF_LE': [b'II*\x00'],
    'TIFF_BE': [b'MM\x00*']
}

# Scan timeouts
LIGHT_SCAN_TIMEOUT = 30
FULL_SCAN_TIMEOUT = 120

# Debug settings
DEBUG_MODE = True
DETAILED_ERRORS = True

# Stateless mode
STATELESS_MODE = True

# ===========================
# DIRECTORY CREATION
# ===========================

# Create required directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SECURITY_DIR, exist_ok=True)
os.makedirs(YARA_RULES_DIR, exist_ok=True)

# Create malware hash file if it doesn't exist
if not MALWARE_HASH_FILE.exists():
    try:
        with open(MALWARE_HASH_FILE, 'w') as f:
            f.write("# Malware SHA256 hashes (one per line)\n")
            f.write("# Example hashes for testing:\n")
    except Exception:
        pass  # Ignore if we can't create it

print(f"Configuration loaded - Flask-AI + Security Scanner + OpenAI Integration")
print(f"Production Mode - Stable thresholds for reliable performance")