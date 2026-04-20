# Usage Guide

## Running the FastAPI Backend
```bash
uvicorn api.api:app --reload --port 8000
```
The API will be available at `http://127.0.0.1:8000`.

## Running the Next.js Frontend
```bash
cd client-app
npm run dev
```
Open your browser at `http://localhost:3000` to access the dashboard.

## Example API Calls
### Single Prediction
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":55,"gender":1,"chestpain":2,"restingBP":130,"serumcholestrol":250,"fastingbloodsugar":0,"restingrelectro":1,"maxheartrate":150,"exerciseangia":0,"oldpeak":1.0,"slope":2,"noofmajorvessels":0,"target":0}'
```
### Batch Prediction (CSV Upload)
```bash
curl -X POST http://127.0.0.1:8000/predict/upload \
  -F "file=@data/heart.csv"
```
### SHAP Explanation
```bash
curl -X POST http://127.0.0.1:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"record_id": 123}
```

For more details, see the **API Reference** section in the README.
