FROM python:3.11-slim AS source
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ARG REPO=athenavillarin/ccs-250-final-project
ADD https://github.com/${REPO}.git/info/refs?service=git-upload-pack /tmp/refs
RUN git clone --depth 1 https://github.com/${REPO}.git /app

FROM node:20-slim AS frontend
COPY --from=source /app/frontend /frontend
WORKDIR /frontend
RUN npm ci && VITE_API_URL= npm run build

FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY --from=source --chown=user /app /app
COPY --from=frontend --chown=user /frontend/dist /app/frontend/dist
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r backend/requirements.txt
USER user
ENV HF_HOME=/tmp/hf FLOOD_DEVICE=cpu
RUN python scripts/download_checkpoints.py
EXPOSE 7860
CMD ["uvicorn", "main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "7860"]
