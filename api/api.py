# api/api.py
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
import json
from fastapi.responses import HTMLResponse, StreamingResponse
import html
import io

from api.schemas import SingleInput, BatchInput, PredictResponse, ExplainResponse, ModelInfoResponse
from api.utils import load_pipeline_and_metadata, normalize_input_dict, build_shap_explainer_if_possible
from api.utils import aggregate_contributions

APP_NAME = "Heart Disease Prediction API"
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model_pipeline.joblib"
MODEL_DIR = BASE_DIR / "models"

app = FastAPI(title=APP_NAME)

# CORS (allow all during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
pipeline, metadata = load_pipeline_and_metadata(MODEL_PATH)
feature_names = metadata.get("features", None)
model_name = metadata.get("model_name", None)

# Try to build SHAP explainer for the inner estimator (if tree or svm model)
shap_explainer, explainer_mode = build_shap_explainer_if_possible(pipeline, metadata)

@app.get("/health")
def health():
    return {"status": "ok", "model": model_name}

@app.get("/model-info", response_model=ModelInfoResponse, status_code=status.HTTP_200_OK)
def model_info():
    """
    Return model CV report and evaluation metrics (if available).
    - cv_report -> models/cv_report.json
    - eval_metrics -> models/eval_results/metrics.json
    """
    certification_path = MODEL_DIR / "research_certification.json"
    cv_report_path = MODEL_DIR / "cv_report.json"
    eval_metrics_path = MODEL_DIR / "eval_results" / "metrics.json"

    cv_report = None
    eval_metrics = None

    try:
        # 1. Prefer Unified Research Certification (Gold Room Protocol)
        if certification_path.exists():
            with open(certification_path, "r") as f:
                cert = json.load(f)
                cv_report = cert.get("cv_results")
                # Add extra certification metadata
                messages.append("Source: Unified Research Certification (Gold Room Protocol)")
        elif cv_report_path.exists():
            with open(cv_report_path, "r") as f:
                cv_report = json.load(f)
        else:
            messages.append(f"cv_report.json not found at {cv_report_path}")
    except Exception as e:
        messages.append(f"Failed to load cv_report.json: {str(e)}")

    try:
        if eval_metrics_path.exists():
            with open(eval_metrics_path, "r") as f:
                eval_metrics = json.load(f)
        else:
            messages.append(f"eval_results/metrics.json not found at {eval_metrics_path}")
    except Exception as e:
        messages.append(f"Failed to load eval_results/metrics.json: {str(e)}")

    return ModelInfoResponse(cv_report=cv_report, eval_metrics=eval_metrics, messages=messages or None)

@app.get("/model-info/ui", response_class=HTMLResponse)
def model_info_ui():
    """
    Simple HTML UI that renders models/cv_report.json and models/eval_results/metrics.json,
    and shows a Chart.js bar chart for per-model ROC-AUC (if available).
    """
    certification_path = MODEL_DIR / "research_certification.json"
    cv_report_path = MODEL_DIR / "cv_report.json"
    eval_metrics_path = MODEL_DIR / "eval_results" / "metrics.json"

    # read files if available
    cv_report = None
    eval_metrics = None
    cert_meta = None
    messages = []

    try:
        # 1. Prefer Unified Research Certification
        if certification_path.exists():
            with open(certification_path, "r") as f:
                cert = json.load(f)
                cv_report = cert.get("cv_results")
                cert_meta = {k: v for k, v in cert.items() if k != "cv_results"}
        elif cv_report_path.exists():
            with open(cv_report_path, "r") as f:
                cv_report = json.load(f)
        else:
            messages.append("cv_report.json not found.")
    except Exception as e:
        messages.append(f"Failed to load cv_report.json: {e}")

    try:
        if eval_metrics_path.exists():
            with open(eval_metrics_path, "r") as f:
                eval_metrics = json.load(f)
        else:
            messages.append("eval_results/metrics.json not found.")
    except Exception as e:
        messages.append(f"Failed to load eval_results/metrics.json: {e}")

    # Prepare chart data from cv_report: model -> roc_auc (if available)
    chart_labels = []
    chart_values = []
    if isinstance(cv_report, dict):
        for mname, metrics in cv_report.items():
            # metrics may be dict with keys or a list; try common keys
            val = None
            if isinstance(metrics, dict):
                for candidate_key in ["roc_auc", "auc", "roc", "mean_test_roc_auc", "mean_test_score"]:
                    if candidate_key in metrics:
                        try:
                            val = float(metrics[candidate_key])
                            break
                        except Exception:
                            pass
            if val is not None:
                chart_labels.append(str(mname))
                chart_values.append(val)

    # Helper to render dict as pretty HTML table (two-column: key, value)
    def dict_to_table(d):
        if not isinstance(d, dict):
            return "<pre>{}</pre>".format(html.escape(json.dumps(d, indent=2)))
        rows = []
        for k, v in d.items():
            # if value is a simple scalar, show it; if dict/list show as pretty JSON
            if isinstance(v, (dict, list)):
                val_html = "<pre style='white-space:pre-wrap;margin:0;'>{}</pre>".format(html.escape(json.dumps(v, indent=2)))
            else:
                val_html = html.escape(str(v))
            rows.append(f"<tr><td style='vertical-align:top;padding:6px;border:1px solid #ddd'><strong>{html.escape(str(k))}</strong></td><td style='padding:6px;border:1px solid #ddd'>{val_html}</td></tr>")
        return "<table style='border-collapse:collapse;width:100%;'>{}</table>".format("".join(rows))

    # Build HTML
    html_parts = []
    html_parts.append(f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Model Info — Heart Disease Prediction</title>
      <style>
        body {{ font-family: Inter, Roboto, -apple-system, "Segoe UI", Arial; margin: 24px; color: #222; }}
        h1,h2 {{ color: #111; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .box {{ background: #fff; border: 1px solid #eee; padding: 18px; margin-bottom: 18px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
        pre {{ background:#f7f7f9; padding:10px; border-radius:6px; overflow:auto; }}
        .muted {{ color:#666; font-size:0.95rem; }}
        .small {{ font-size:0.9rem; color:#444; }}
        .flex-row {{ display:flex; gap:12px; align-items:flex-start; }}
        .col {{ flex:1; min-width: 260px; }}
        .msg {{ color:#b33; }}
        .badge {{ display:inline-block; background:#eef6ff; color:#0077cc; padding:4px 8px; border-radius:999px; font-size:0.85rem; margin-left:8px; }}
        .chart-wrap {{ max-width:800px; margin: 12px auto 0; }}
      </style>
      <!-- Chart.js CDN -->
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
      <div class="container">
        <h1>Model Info <span class="badge">Heart Disease Prediction</span></h1>
        <p class="muted">This page shows CV results, evaluation metrics and a quick chart comparing models (ROC-AUC) for the trained models saved under <code>models/</code>.</p>
    """)

    # Certification Banner
    if cert_meta:
        html_parts.append(f"""
        <div class="box" style="border-left: 5px solid #0077cc; background: #f0f7ff;">
          <h2 style="margin-top:0; color:#0055aa;">🛡️ Research-Grade Certification</h2>
          <div class="flex-row" style="flex-wrap:wrap;">
            <div class="col"><p class="small"><strong>Status:</strong> {cert_meta.get('status', 'N/A')}</p></div>
            <div class="col"><p class="small"><strong>Isolation:</strong> {cert_meta.get('gold_samples', 'N/A')} Gold Samples (Strictly Unseen)</p></div>
            <div class="col"><p class="small"><strong>Development:</strong> {cert_meta.get('dev_samples', 'N/A')} Samples (10-fold CV)</p></div>
            <div class="col"><p class="small"><strong>Clinical Standard:</strong> {cert_meta.get('clinical_standard', 'N/A')}</p></div>
          </div>
        </div>
        """)

    # messages
    if messages:
        html_parts.append('<div class="box"><h2>Messages</h2>')
        for m in messages:
            html_parts.append(f'<p class="msg">{html.escape(m)}</p>')
        html_parts.append('</div>')

    # Chart area (only if chart data present)
    html_parts.append('<div class="box"><h2>Model comparison</h2>')
    if chart_labels and chart_values:
        # embed canvas and a small legend
        html_parts.append("""
            <div class="chart-wrap">
              <canvas id="cvChart" width="800" height="320"></canvas>
            </div>
            <script>
              const labels = %s;
              const values = %s;
              const ctx = document.getElementById('cvChart').getContext('2d');
              new Chart(ctx, {
                type: 'bar',
                data: {
                  labels: labels,
                  datasets: [{
                    label: 'CV ROC-AUC',
                    data: values,
                    borderWidth: 1,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)'
                  }]
                },
                options: {
                  scales: {
                    y: { beginAtZero: true, max: 1.0 }
                  },
                  plugins: {
                    legend: { display: false }
                  }
                }
              });
            </script>
        """ % (json.dumps(chart_labels), json.dumps(chart_values)))
    else:
        html_parts.append('<p class="small">Not enough CV data to show comparison chart (need roc_auc values in cv_report.json).</p>')
    html_parts.append('</div>')

    # CV report
    html_parts.append('<div class="box"><h2>Cross-validation report (cv_report.json)</h2>')
    if cv_report:
        # If cv_report is a dict of models -> metrics
        if isinstance(cv_report, dict):
            for model_name, metrics in cv_report.items():
                html_parts.append(f'<div style="margin-bottom:10px;"><h3 style="margin:6px 0;">{html.escape(model_name)}</h3>')
                html_parts.append(dict_to_table(metrics))
                html_parts.append('</div>')
        else:
            html_parts.append(dict_to_table(cv_report))
    else:
        html_parts.append('<p class="small">No CV report available.</p>')
    html_parts.append('</div>')

    # Eval metrics
    html_parts.append('<div class="box"><h2>Evaluation metrics (models/eval_results/metrics.json)</h2>')
    if eval_metrics:
        html_parts.append(dict_to_table(eval_metrics))
    else:
        html_parts.append('<p class="small">No evaluation metrics available.</p>')
    html_parts.append('</div>')

    # raw JSON toggles
    html_parts.append("""
      <div class="box">
        <h2>Raw JSON</h2>
        <div class="flex-row">
          <div class="col"><h4>cv_report.json</h4><pre>{cv}</pre></div>
          <div class="col"><h4>metrics.json</h4><pre>{metrics}</pre></div>
        </div>
      </div>
    """.format(
        cv=html.escape(json.dumps(cv_report, indent=2)) if cv_report is not None else "null",
        metrics=html.escape(json.dumps(eval_metrics, indent=2)) if eval_metrics is not None else "null"
    ))

    # footer
    html_parts.append("""
        <div style="text-align:center; margin-top:18px; color:#666; font-size:0.9rem;">
          API endpoints: <code>/predict</code>, <code>/predict/batch</code>, <code>/predict/upload</code>, <code>/explain</code>, <code>/model-info</code>
        </div>
      </div>
    </body>
    </html>
    """)

    return HTMLResponse("".join(html_parts))


@app.post("/predict", response_model=PredictResponse)
def predict_single(payload: SingleInput):
    try:
        d = payload.model_dump()
        d = normalize_input_dict(d)
        # build DataFrame in training order if possible
        if feature_names:
            X = pd.DataFrame([d], columns=feature_names)
        else:
            X = pd.DataFrame([d])
        probs = pipeline.predict_proba(X)[:, 1]
        preds = pipeline.predict(X)
        return PredictResponse(prediction=int(preds[0]), probability=float(probs[0]))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
def predict_batch(payload: BatchInput):
    try:
        rows = [normalize_input_dict(r.dict()) for r in payload.records]
        if feature_names:
            X = pd.DataFrame(rows, columns=feature_names)
        else:
            X = pd.DataFrame(rows)
        probs = pipeline.predict_proba(X)[:, 1].tolist()
        preds = pipeline.predict(X).astype(int).tolist()
        return {"predictions": preds, "probabilities": probs}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...), download: bool = Query(False)):
    """
    Upload CSV file with the same column names as training features.
    If download=true, returns a CSV file with appended 'prediction' and 'probability' columns.
    Otherwise returns JSON: { predictions: [...], probabilities: [...] }.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        # optional: validate columns presence
        if feature_names:
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing columns in uploaded CSV: {missing}")
            X = df[feature_names]
        else:
            X = df

        probs = pipeline.predict_proba(X)[:, 1].tolist()
        preds = pipeline.predict(X).astype(int).tolist()

        if download:
            df_out = df.copy()
            df_out["prediction"] = preds
            df_out["probability"] = probs
            buf = io.BytesIO()
            df_out.to_csv(buf, index=False)
            buf.seek(0)
            return StreamingResponse(buf, media_type="text/csv",
                                     headers={"Content-Disposition": "attachment; filename=predictions.csv"})
        else:
            return {"predictions": preds, "probabilities": probs}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain", response_model=ExplainResponse)
def explain_single(payload: SingleInput):
    """
    Returns prediction + probability + per-feature SHAP contributions (if SHAP explainer available).
    For large models/explainers this may be slower — suitable for single-sample explanations.
    """
    try:
        d = payload.model_dump()
        d = normalize_input_dict(d)
        if feature_names:
            X = pd.DataFrame([d], columns=feature_names)
        else:
            X = pd.DataFrame([d])
        probs = pipeline.predict_proba(X)[:, 1]
        preds = pipeline.predict(X)

        base_value = None
        contributions = None
        aggregated = None
        top_features = None

        # If explainer available compute shap values for the inner model.
        if shap_explainer is not None:
            # Need to transform input through preprocessor (if present) before passing to explainer
            pre = pipeline.named_steps.get("preprocessor", None)
            if pre is not None:
                X_trans = pre.transform(X)
            else:
                X_trans = X.values
            # shap_explainer expects array-like transformed features
            if explainer_mode == "kernel":
                shap_vals = shap_explainer.shap_values(X_trans, nsamples=100)
            else:
                shap_vals = shap_explainer.shap_values(X_trans)
            # Normalize shap_vals to a 1D array for the current sample
            if isinstance(shap_vals, list):
                # pick positive class (index 1) when available
                shap_arr = np.array(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0])
            else:
                shap_arr = np.array(shap_vals)

            # Defensive squeeze and index to get 1D array for the first sample
            if shap_arr.ndim > 2:
                # e.g. (n_samples, n_features, n_classes) -> pick class 1 and first sample
                if shap_arr.shape[-1] > 1:
                    shap_arr = shap_arr[..., 1]
                shap_arr = np.squeeze(shap_arr)
            
            if shap_arr.ndim == 2:
                sv = shap_arr[0]
            else:
                sv = shap_arr # already 1D
            
            # Final cast to ensure we have a list of floats
            sv = [float(x) for x in np.atleast_1d(sv)]

            # compute base value if available on explainer
            try:
                if hasattr(shap_explainer, "expected_value"):
                    ev = shap_explainer.expected_value
                    if isinstance(ev, (list, np.ndarray)):
                        try:
                            base_value = float(np.atleast_1d(ev)[1])
                        except Exception:
                            base_value = float(np.atleast_1d(ev)[0])
                    else:
                        base_value = float(ev)
            except Exception:
                base_value = None

            # map feature names after transformation if possible
            transformed_names = metadata.get("transformed_feature_names", None)
            if transformed_names is None:
                try:
                    transformed_names = pre.get_feature_names_out(feature_names).tolist()
                except Exception:
                    transformed_names = list(X.columns)

            # pair names with contributions
            contributions = dict(zip(transformed_names, [float(x) for x in sv]))

            # aggregate contributions back to original feature names (sum one-hot pieces)
            try:
                aggregated_dict, top_list = aggregate_contributions(contributions, top_k=8)
                aggregated = aggregated_dict
                # prepare top_features as list of dicts for response
                top_features = [{"feature": name, "impact": float(val)} for name, val in top_list]
            except Exception:
                aggregated = None
                top_features = None

        return ExplainResponse(
            prediction=int(preds[0]),
            probability=float(probs[0]),
            contributions=contributions,
            base_value=base_value,
            aggregated=aggregated,
            top_features=top_features
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
