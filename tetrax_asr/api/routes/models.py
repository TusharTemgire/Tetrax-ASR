"""
Models route: GET /v1/models
"""

from fastapi import APIRouter, Request
from tetrax_asr.api.schemas.audio import ModelsListResponse, ModelInfo
from tetrax_asr.config import model_config

router = APIRouter(prefix="/v1", tags=["Models"])


@router.get("/models", response_model=ModelsListResponse)
async def list_models(request: Request):
    engine = getattr(request.app.state, "engine", None)

    is_finetuned = engine.is_finetuned if engine else False
    active_name = "tetrax-stt-indian-en" if is_finetuned else f"{model_config.base_model_id} (base)"

    models = [
        ModelInfo(
            id="tetrax-stt-indian-en",
            name="Tetrax Indian-English Speech Intelligence (LoRA)",
            base_checkpoint=model_config.base_model_id,
            is_finetuned=is_finetuned,
            adapter_path=model_config.adapter_path,
            language_focus="English (Indian Accents / Svarah adapted)",
            architecture="Whisper Transformer + PEFT LoRA",
            device=model_config.device,
            compute_type=model_config.compute_type
        ),
        ModelInfo(
            id="openai/whisper-small",
            name="Whisper Small Baseline (OpenAI)",
            base_checkpoint="openai/whisper-small",
            is_finetuned=False,
            adapter_path=None,
            language_focus="Multilingual",
            architecture="Whisper Transformer (244M params)",
            device=model_config.device,
            compute_type=model_config.compute_type
        ),
        ModelInfo(
            id="openai/whisper-base",
            name="Whisper Base Baseline (OpenAI)",
            base_checkpoint="openai/whisper-base",
            is_finetuned=False,
            adapter_path=None,
            language_focus="Multilingual",
            architecture="Whisper Transformer (74M params)",
            device=model_config.device,
            compute_type=model_config.compute_type
        ),
    ]

    return ModelsListResponse(
        active_model=active_name,
        models=models
    )
