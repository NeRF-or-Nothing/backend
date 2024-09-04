# syntax=docker/dockerfile:experimental

# Build stage
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3.8 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY submodules/simple-knn submodules/simple-knn
COPY submodules/diff-gaussian-rasterization submodules/diff-gaussian-rasterization

RUN --mount=type=cache,target=/root/.cache/pip \
    cd submodules/simple-knn && \
    python3.8 setup.py bdist_wheel && \
    cd ../diff-gaussian-rasterization && \
    python3.8 setup.py bdist_wheel

# Final stage
FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY --from=builder /build/submodules/simple-knn/dist/*.whl .
COPY --from=builder /build/submodules/diff-gaussian-rasterization/dist/*.whl .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install *.whl && rm *.whl

COPY . .