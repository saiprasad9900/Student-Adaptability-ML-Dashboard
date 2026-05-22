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

## GitHub deployment

To publish this project to GitHub, run the following from the project root:

```bash
cd c:\Users\dell\OneDrive\Documents\Major_Project

git init
git add .
git commit -m "Initial commit"
```

Then create a repository on GitHub and add the remote URL. Replace `USERNAME` and `REPO` with your values:

```bash
git remote add origin https://github.com/USERNAME/REPO.git
git branch -M main
git push -u origin main
```

## Optional: Deploy to Streamlit Cloud

1. Push the repository to GitHub.
2. Go to https://streamlit.io/cloud and connect your GitHub account.
3. Select this repository and deploy the app.

## Notes

- The `outputs/` directory is ignored in `.gitignore` because it contains generated files.
- If you want the app to work after cloning, keep `best_model.joblib` in the repository or retrain the model locally.
