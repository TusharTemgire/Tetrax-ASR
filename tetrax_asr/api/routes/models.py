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
    active_name = "tetrax-stt-marathi-hindi" if is_finetuned else f"{model_config.base_model_id} (base)"

    models = [
        ModelInfo(
            id="tetrax-stt-marathi-hindi",
            name="Tetrax Marathi & Hindi Speech Intelligence (Whisper-Medium + LoRA)",
            base_checkpoint=model_config.base_model_id,
            is_finetuned=is_finetuned,
            adapter_path=model_config.adapter_path,
            language_focus="Marathi ('mr') & Hindi ('hi') - IndicVoices Adapted",
            architecture="Whisper Medium Transformer + PEFT LoRA (769M params)",
            device=model_config.device,
            compute_type=model_config.compute_type
        ),
        ModelInfo(
            id="openai/whisper-medium",
            name="Whisper Medium Multilingual Baseline (OpenAI)",
            base_checkpoint="openai/whisper-medium",
            is_finetuned=False,
            adapter_path=None,
            language_focus="Multilingual (99+ languages)",
            architecture="Whisper Transformer (769M params)",
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
    ]

    return ModelsListResponse(
        active_model=active_name,
        models=models
    )
