# Insurance-prediction-pipeline
This project demonstrates practical ML deployment and containerization for a real-world regression problem.

Insurance Charges Prediction Model

A complete end-to-end Machine Learning project that predicts insurance charges based on demographic and medical attributes.
This repository includes data preprocessing, model training, evaluation, deployment script, and Docker support for containerized execution.

📌 About

This project builds a regression model that predicts an individual’s insurance charges using attributes such as age, BMI, number of children, smoking habits, and more. It demonstrates how machine learning can be used to create cost-estimation tools for insurance companies, hospitals, and health analytics platforms.

The project includes:
Full ML pipeline
Model serialization
Inference-ready application script
Dockerized environment for easy deployment anywhere

🧠 Model Overview

The notebook (insurance.ipynb) covers:

✔ 1. Exploratory Data Analysis

Checking missing values
Distribution plots
Outlier checks
Correlation analysis
Categorical vs numerical variable exploration

✔ 2. Data Preprocessing

Label encoding categorical variables
Converting smoker/sex/region into numeric
Scaling numerical features
Train-test split

✔ 3. Model Training

Linear Regression


👉 The best-performing model is exported as:
model.pkl
scaler.pkl
columns.pkl

✔ 4. Evaluation

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score

Residual Analysis

🗂 Project Structure
📦 insurance-prediction
├── insurance.ipynb
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md

⚡ How to Run Locally
1. Install dependencies
pip install -r requirements.txt

python app.py

🐳 Docker Support

This project is fully dockerized, so you can run it without installing Python or dependencies.

🧱 Dockerfile (included in repo)

Typical structure (modify based on your setup):

FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

🚀 Run the Project in Docker
1. Build the Docker image
docker build -t insurance-prediction .

2. Run the container
docker run -p 8501:8501 insurance-prediction


Now open:
👉 http://localhost:8501
to use your ML prediction app.

📈 Model Prediction Flow

User Input → Preprocessing → Scaler → Model → Predicted Insurance Charge
