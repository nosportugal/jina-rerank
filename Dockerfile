# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

ARG PYTHON_VERSION=3.13
ARG COMPUTE_DEVICE=cpu

# --- Builder Stage ---
FROM python:${PYTHON_VERSION}-slim AS builder

ARG COMPUTE_DEVICE

WORKDIR /src

# Install uv and its dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN chmod +x /bin/uv /bin/uvx && uv venv .venv
ENV PATH="/src/.venv/bin:$PATH"

# Copy dependency specification and install production dependencies
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-default-groups $( [ "$COMPUTE_DEVICE" = "gpu" ] && echo "--group gpu" )

# For GPU builds, completely remove onnxruntime and install onnxruntime-gpu
RUN if [ "$COMPUTE_DEVICE" = "gpu" ]; then \
    uv pip uninstall onnxruntime onnxruntime-gpu || true && \
    rm -rf .venv/lib/python*/site-packages/onnxruntime* && \
    uv pip install --force-reinstall --no-deps onnxruntime-gpu; \
    fi


# --- Final Image (CPU) ---
FROM python:${PYTHON_VERSION}-slim AS final-cpu

ARG PORT=80
ARG COMPUTE_DEVICE=cpu
ARG VERSION
ARG BUILD_ID
ARG COMMIT_SHA

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=${PORT}
ENV COMPUTE_DEVICE=${COMPUTE_DEVICE}
ENV VERSION=${VERSION}
ENV BUILD_ID=${BUILD_ID}
ENV COMMIT_SHA=${COMMIT_SHA}

WORKDIR /src

# Create a non-root user
RUN addgroup --system app && adduser --system --group --no-create-home app

# Create the model directory
RUN mkdir -p /src/.model && chown app:app /src/.model

# Copy only the needed virtual environment from builder
COPY --from=builder --chown=app:app /src/.venv .venv
ENV PATH="/src/.venv/bin:$PATH"

# Copy the application code
COPY --chown=app:app main.py /src/main.py
COPY --chown=app:app models /src/models

# Use the non-root user
USER app:app

RUN python -m main download

# https://cloud.google.com/run/docs/tips/python#optimize_gunicorn
EXPOSE $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info --timeout-keep-alive 0"]


# --- Final Image (GPU) ---
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS final-gpu

ARG PYTHON_VERSION=3.13
ARG PORT=80
ARG COMPUTE_DEVICE=gpu
ARG VERSION
ARG BUILD_ID
ARG COMMIT_SHA

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=${PORT}
ENV COMPUTE_DEVICE=${COMPUTE_DEVICE}
ENV VERSION=${VERSION}
ENV BUILD_ID=${BUILD_ID}
ENV COMMIT_SHA=${COMMIT_SHA}
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /src

# Install Python 3.13 from deadsnakes PPA (Ubuntu 22.04 only has Python 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Create a non-root user
RUN addgroup --system app && adduser --system --group --no-create-home app

# Create the model directory
RUN mkdir -p /src/.model && chown app:app /src/.model

# Copy only the needed virtual environment from builder
COPY --from=builder --chown=app:app /src/.venv .venv

# Fix venv symlinks to point to the correct Python version in this image
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /src/.venv/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /src/.venv/bin/python3 && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /src/.venv/bin/python${PYTHON_VERSION}

ENV PATH="/src/.venv/bin:$PATH"

# Copy the application code
COPY --chown=app:app main.py /src/main.py
COPY --chown=app:app models /src/models

# Use the non-root user
USER app:app

RUN python -m main download

# https://cloud.google.com/run/docs/tips/python#optimize_gunicorn
EXPOSE $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info --timeout-keep-alive 0"]


# --- Default target based on COMPUTE_DEVICE ---
ARG COMPUTE_DEVICE=cpu
FROM final-${COMPUTE_DEVICE} AS final