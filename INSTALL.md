# Installation Guide

## Prerequisites
- **Python** >= 3.9
- **Node.js** >= 18
- **Git**
- **Virtualenv** (optional but recommended)

## Steps
```bash
# Clone the repository
git clone https://github.com/yourusername/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Install frontend dependencies
cd client-app
npm install
cd ..
```

You are now ready to run the backend and frontend as described in the **Usage** section of the README.
