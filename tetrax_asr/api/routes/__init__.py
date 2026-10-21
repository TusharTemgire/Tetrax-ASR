from .transcriptions import router as transcriptions_router
from .stream import router as stream_router
from .models import router as models_router
from .health import router as health_router

__all__ = [
    "transcriptions_router",
    "stream_router",
    "models_router",
    "health_router",
]
