# AI-Powered Water Quality Monitoring System

This project is a complete Streamlit application that predicts whether water is safe or unsafe for drinking using a machine learning model trained on water quality parameters.

## Features
- Modern Streamlit dashboard with sidebar navigation
- Data exploration and analytics pages
- Potability prediction with confidence and risk level
- Local prediction history tracking
- PDF report export and CSV history export
- Model training and persistence via joblib

## Project Structure
- app.py: Main app entry point
- train_model.py: Model training pipeline
- predict.py: Prediction helpers
- pages/: Home, Prediction, Dashboard, About
- utils/: Reusable preprocessing, visualization, and helper utilities
- data/: Input dataset CSV
- model/: Trained model artifact

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python train_model.py
   ```
3. Launch the app:
   ```bash
   streamlit run app.py
   ```

## Deployment on Streamlit Community Cloud
1. Push this project to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Select the repository and branch.
4. Set the main file path to app.py.
5. Deploy.

## Notes
The app includes a sample water potability dataset in the data folder so it works out of the box.
