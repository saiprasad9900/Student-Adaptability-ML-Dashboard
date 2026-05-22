# Student Adaptability ML Dashboard

This project is a Streamlit dashboard for analyzing student adaptability data, visualizing model performance, and running predictions.

## Files

- `streamlit_app.py` — Streamlit frontend application
- `student_adaptability_ml.py` — Data processing and model training utilities
- `students_adaptability_data.csv` — Default dataset
- `best_model.joblib` — Saved model bundle
- `outputs/` — Generated output files and plots

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## GitHub repository

**Live repo:** https://github.com/saiprasad9900/Student-Adaptability-ML-Dashboard

## Deploy to Streamlit Cloud (live web app)

1. Open [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** → **From existing repo**.
3. Select **saiprasad9900/Student-Adaptability-ML-Dashboard**, branch **main**.
4. Set **Main file path** to `streamlit_app.py`, then click **Deploy**.

After deployment, Streamlit gives you a public URL (for example `https://student-adaptability-ml-dashboard.streamlit.app`).

## Notes

- The `outputs/` directory is ignored in `.gitignore` because it contains generated files.
- If you want the app to work after cloning, keep `best_model.joblib` in the repository or retrain the model locally.
