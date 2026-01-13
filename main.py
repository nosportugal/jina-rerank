#  Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
#  Permission is hereby granted, free of charge, to any person
#  obtaining a copy of this software and associated documentation
#  files (the "Software"), to deal in the Software without
#  restriction, subject to the conditions in the full MIT License.
#  The Software is provided "as is", without warranty of any kind.

import os
import time
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import RedirectResponse
from fastembed.rerank.cross_encoder import TextCrossEncoder
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from models import (
    RerankRequest,
    RerankResult,
    RerankResponse,
    InfoResponse,
)

##
# Load the config
##
MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
VERSION = os.getenv("VERSION") or "unknown"
BUILD_ID = os.getenv("BUILD_ID") or "unknown"
COMMIT_SHA = os.getenv("COMMIT_SHA") or "unknown"
PORT = int(os.getenv("PORT") or "8000")
COMPUTE_DEVICE = os.getenv("COMPUTE_DEVICE") or "cpu"
DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N") or "0")  # 0 means return all
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

##
# Configure logging
##
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

##
# Prometheus Metrics
##
REQUEST_COUNT = Counter(
    "jina_rerank_requests_total",
    "Total number of rerank requests",
    ["compute_device", "status"]
)

REQUEST_LATENCY = Histogram(
    "jina_rerank_request_latency_seconds",
    "Request latency in seconds",
    ["compute_device"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

RERANK_LATENCY = Histogram(
    "jina_rerank_inference_latency_seconds",
    "Inference latency in seconds (model computation only)",
    ["compute_device"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

DOCUMENTS_PER_REQUEST = Histogram(
    "jina_rerank_documents_per_request",
    "Number of documents per rerank request",
    ["compute_device"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

MODEL_LOADED = Gauge(
    "jina_rerank_model_loaded",
    "Whether the model is loaded (1) or not (0)",
    ["compute_device", "model_name"]
)


##
# Create the FastAPI app
##
app = FastAPI(
    title="Multilingual Reranker API",
    description=f"API for reranking documents based on query relevance using {MODEL_NAME}",
    version=VERSION,
)

##
# Load the model (deferred to startup)
##
reranker = None


@app.on_event("startup")
async def load_model():
    global reranker
    try:
        logger.info(f"Loading model {MODEL_NAME}...")
        
        # Check if GPU is requested and available
        if COMPUTE_DEVICE == "gpu":
            try:
                import torch
                use_gpu = torch.cuda.is_available()
                logger.info(f"GPU Available: {use_gpu}")
                if use_gpu:
                    logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
                    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            except ImportError:
                logger.warning("Torch not available, GPU detection skipped")
                use_gpu = False
        else:
            use_gpu = False
        
        reranker = TextCrossEncoder(
            model_name=MODEL_NAME,
            cache_dir=str(Path(__file__).parent.absolute() / ".model"),
            providers=["CUDAExecutionProvider"] if COMPUTE_DEVICE == "gpu" else ["CPUExecutionProvider"],
        )
        
        MODEL_LOADED.labels(compute_device=COMPUTE_DEVICE, model_name=MODEL_NAME).set(1)
        logger.info(f"Model {MODEL_NAME} loaded successfully on {COMPUTE_DEVICE.upper()}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")


##
# Routes
##
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/info", response_model=InfoResponse)
async def info():
    return InfoResponse()


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest = Body(...)):
    start_time = time.time()
    
    if reranker is None:
        REQUEST_COUNT.labels(compute_device=COMPUTE_DEVICE, status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        # Track documents per request
        DOCUMENTS_PER_REQUEST.labels(compute_device=COMPUTE_DEVICE).observe(len(request.documents))
        
        # Determine top_n: use request value, fall back to env var, or None for all
        top_n = request.top_n if request.top_n is not None else (DEFAULT_TOP_N if DEFAULT_TOP_N > 0 else None)
        
        logger.debug(f"Rerank request: query='{request.query[:50]}...', documents={len(request.documents)}, top_n={top_n} (requested={request.top_n}, default={DEFAULT_TOP_N}), batch_size={request.batch_size}")
        
        # Compute the relevance scores (measure inference time separately)
        inference_start = time.time()
        scores = list(
            reranker.rerank(
                request.query,
                documents=request.documents,
                batch_size=request.batch_size,
                top_n=top_n,
            )
        )
        inference_duration = time.time() - inference_start
        RERANK_LATENCY.labels(compute_device=COMPUTE_DEVICE).observe(inference_duration)

        logger.debug(f"Inference completed in {inference_duration:.4f}s, scores={scores[:3]}...")

        # Build results with index and relevance_score
        results = [
            RerankResult(index=i, relevance_score=score)
            for i, score in enumerate(scores)
        ]

        # Record success metrics
        REQUEST_COUNT.labels(compute_device=COMPUTE_DEVICE, status="success").inc()
        total_duration = time.time() - start_time
        REQUEST_LATENCY.labels(compute_device=COMPUTE_DEVICE).observe(total_duration)

        logger.debug(f"Request completed in {total_duration:.4f}s")

        response = RerankResponse(
            model=MODEL_NAME,
            object="list",
            results=results,
        )
        
        logger.debug(f"Response: {response.model_dump_json()}")

        return response

    except Exception as e_:
        REQUEST_COUNT.labels(compute_device=COMPUTE_DEVICE, status="error").inc()
        REQUEST_LATENCY.labels(compute_device=COMPUTE_DEVICE).observe(time.time() - start_time)
        logger.error(f"Error during reranking: {str(e_)}")
        raise HTTPException(
            status_code=500, detail=f"Error during reranking: {str(e_)}"
        ) from e_


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "serve"

    # Start the server
    if command == "serve":
        import uvicorn

        uvicorn.run("main:app", host="0.0.0.0", port=PORT)

    # Download the model
    elif command == "download":
        sys.exit(0)
