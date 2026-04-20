# API Reference

The FastAPI backend provides a set of RESTful endpoints for prediction, batch processing, and model interpretation.

## Endpoints
| Path | Method | Description | Request Body | Response |
|------|--------|-------------|--------------|----------|
| `/predict` | POST | Single patient prediction | JSON with all 14 features (see table below) | `{ "prediction": 0, "probability": 0.87 }` |
| `/predict/batch` | POST | Batch predictions for a list of records | JSON array of feature objects | `{ "predictions": [{"prediction":0,"probability":0.87}, ...] }` |
| `/predict/upload` | POST | CSV file upload for bulk predictions | `multipart/form-data` with `file` field | Same as batch response |
| `/explain` | POST | SHAP explanation for a specific record | `{ "record_id": 123 }` (or full feature dict) | `{ "shap_values": [...], "base_value": 0.5 }` |
| `/model-info` | GET | Returns JSON with model metadata, CV scores, hyper‑parameters | – | `{ "model": "LightGBM", "accuracy": 0.985, "roc_auc": 0.99, ... }` |
| `/model-info/ui` | GET | Returns an HTML dashboard with performance plots | – | HTML page (served from `models/comparison_plots/`) |

## Feature Schema
| Feature | Type | Description |
|---------|------|-------------|
| `age` | int | Age in years |
| `gender` | int (0/1) | 1 = Male, 0 = Female |
| `chestpain` | int (1‑4) | Chest pain type (1: Typical, 2: Atypical, 3: Non‑anginal, 4: Asymptomatic) |
| `restingBP` | int | Resting blood pressure (mm Hg) |
| `serumcholestrol` | int | Serum cholesterol (mg/dl) |
| `fastingbloodsugar` | int (0/1) | Fasting blood sugar > 120 mg/dl |
| `restingrelectro` | int (0‑2) | Resting ECG results |
| `maxheartrate` | int | Maximum heart rate achieved |
| `exerciseangia` | int (0/1) | Exercise‑induced angina |
| `oldpeak` | float | ST depression induced by exercise |
| `slope` | int (1‑3) | Slope of the ST segment |
| `noofmajorvessels` | int (0‑3) | Number of major vessels (colored by fluoroscopy) |
| `target` | int (0/1) | Diagnosis (0 = Normal, 1 = Disease) |

All endpoints return standard HTTP status codes (`200` for success, `400` for validation errors, `500` for server errors). Errors include a JSON payload with `detail` field.

## Error Example
```json
{ "detail": "Missing required field: age" }
```

For more detailed usage, refer to the **Usage Guide** (`USAGE.md`).
