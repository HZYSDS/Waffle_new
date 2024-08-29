FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.8=3.8.5* \
    python3-pip=20.3* \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip==20.3 \
    && pip install \
    numpy==1.23.1 \
    scipy==1.10.0 \
    tqdm \
    webdataset==0.2.5 \
    torchmetrics==0.6.0 \
    matplotlib \
    termcolor \
    git+https://github.com/openai/CLIP.git \
    git+https://github.com/modestyachts/ImageNetV2_pytorch \
    -e .

COPY . /workspace

CMD ["python3"]