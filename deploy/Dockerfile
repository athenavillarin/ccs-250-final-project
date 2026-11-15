FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN VITE_API_URL= npm run build

FROM python:3.11-slim
WORKDIR /app
COPY deploy/requirements.txt deploy/requirements.txt
RUN pip install --no-cache-dir -r deploy/requirements.txt
COPY src src
COPY scripts/download_checkpoints.py scripts/download_checkpoints.py
RUN python scripts/download_checkpoints.py --onnx
COPY results/main_results.csv results/main_results.csv
COPY backend/main.py backend/main.py
COPY --from=frontend /frontend/dist frontend/dist
ENV FLOOD_RUNTIME=onnx FLOOD_THREADS=1
CMD ["sh", "-c", "uvicorn main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-10000}"]
