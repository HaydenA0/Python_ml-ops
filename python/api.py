"""ASGI entrypoints for the FastAPI services.

Usage:
    uvicorn python.api:app              -- single ResNet18 model
    uvicorn python.api:ensemble_app     -- stacked ensemble
"""

from python.pcxp_mlops.api import app as _app, create_app, create_ensemble_app

app = _app
ensemble_app = create_ensemble_app()

__all__ = ["app", "ensemble_app", "create_app", "create_ensemble_app"]
