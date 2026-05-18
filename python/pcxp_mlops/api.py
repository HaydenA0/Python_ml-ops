"""FastAPI application exposing the trained model."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .config import PROJECT_VERSION
from .predict import InferenceService


class PredictRequest(BaseModel):
    """Prediction request payload."""

    image_path: str | None = None
    image_base64: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_input_source(self):
        if bool(self.image_path) == bool(self.image_base64):
            raise ValueError("Provide exactly one of image_path or image_base64.")
        return self


def create_app(service: InferenceService | None = None) -> FastAPI:
    """Create the API application with an optional injected inference service."""
    inference_service = service or InferenceService()
    static_dir = Path(__file__).resolve().parent / "static"
    app = FastAPI(
        title="PCXP Pneumonia Prediction API",
        version=PROJECT_VERSION,
        description="REST API for chest X-ray pneumonia classification.",
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health():
        return inference_service.health()

    @app.get("/model-info")
    def model_info():
        return inference_service.model_info()

    @app.post("/predict")
    def predict(request: PredictRequest):
        try:
            result = inference_service.predict(
                image_path=request.image_path,
                image_base64=request.image_base64,
                threshold=request.threshold,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Prediction failed.") from exc

        return {
            "request_id": str(uuid4()),
            "predicted_class": result.predicted_class,
            "predicted_index": result.predicted_index,
            "probability": result.probability,
            "threshold": result.threshold,
            "model_version": result.model_version,
        }

    return app


app = create_app()
