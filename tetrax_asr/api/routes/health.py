"""
Health route: GET /v1/health
"""

import time
import psutil
import torch
from fastapi import APIRouter, Request

router = APIRouter(prefix="/v1", tags=["Health"])

START_TIME = time.time()


@router.get("/health")
async def health_check(request: Request):
    engine = getattr(request.app.state, "engine", None)
    engine_ready = engine is not None

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2)
        }
    else:
        gpu_info = {"available": False, "device": "cpu"}

    vm = psutil.virtual_memory()

    return {
        "status": "healthy" if engine_ready else "degraded",
        "engine_ready": engine_ready,
        "is_finetuned_adapter": engine.is_finetuned if engine_ready else False,
        "active_model": engine.base_model_id if engine_ready else None,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "system_ram": {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "percent_used": vm.percent
        },
        "gpu": gpu_info
    }
