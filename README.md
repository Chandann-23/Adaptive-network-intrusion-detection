---
title: Adaptive Network Intrusion Detection System
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Zero-day network attack detection via hybrid ML pipeline
---

# Adaptive NIDS — Hybrid Zero-Day Detection API

A production-grade **Network Intrusion Detection System** built on the NSL-KDD dataset.

## Architecture

```
Input (41 network features)
        ↓
Stage 1: Isolation Forest (novelty detection)
        ↓ anomaly flagged?
Stage 2: XGBoost Multiclass Router
        ↓
Prediction: Normal / DoS / Probe / R2L / U2R
```

## Key Results

| Metric | Value |
|---|---|
| Novel Attack Recall | **89.4%** |
| Seen Attack Recall | 83.1% |
| False Positive Rate | 12.7% |
| Architecture | Two-stage hybrid (IF + XGBoost) |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info & version |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/predict` | Single connection prediction |
| `POST` | `/predict_batch` | Batch prediction |

## Sample Request

```bash
curl -X POST https://<your-space>.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 232,
    "dst_bytes": 8153,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 5,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 5,
    "srv_count": 5,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 255,
    "dst_host_srv_count": 255,
    "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.01,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0
  }'
```

## Sample Response

```json
{
  "is_attack": false,
  "attack_type": "normal",
  "attack_family": "Normal Traffic",
  "confidence": 0.94,
  "anomaly_score": -0.12,
  "stage1_decision": "normal",
  "processing_time_ms": 2.3
}
```

## Tech Stack

- **Model**: Isolation Forest (Stage 1) + XGBoost (Stage 2)
- **Dataset**: NSL-KDD (125,973 training samples)
- **API**: FastAPI + Uvicorn
- **Container**: Docker (python:3.10-slim)
- **Frontend**: [Vercel Dashboard](#)
