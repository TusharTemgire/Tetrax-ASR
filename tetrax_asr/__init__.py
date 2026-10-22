"""
Tetrax ASR - Real-Time Speech Intelligence
"""

import os
import logging

# Ensure Hugging Face operates in pure offline mode from local disk cache
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Silence noisy HTTP connection and cache checking logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

__version__ = "0.1.0"
