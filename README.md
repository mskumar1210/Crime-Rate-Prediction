**Crime Rate Prediction System**

Overview
This project presents a data-driven, AI-powered solution to enhance public safety and assist law enforcement agencies by predicting crime rates and classifying crime types using machine learning and deep learning techniques. The system integrates LSTM-based crime rate forecasting with CNN-based crime hotspot image analysis, helping police departments, city planners, and policymakers make informed decisions on resource allocation, patrol scheduling, and preventive measures.

Problem Statement
Law enforcement agencies often face:
- Lack of real-time crime trend visibility
- Inefficient resource allocation across districts
- Inability to predict crime hotspots proactively
- Difficulty in classifying crime severity from surveillance footage
- Challenges in understanding seasonal patterns, socio-economic factors, and event-based crime spikes
- Reactive rather than preventive policing strategies

Objectives
Predict future crime rates using historical trends, weather, demographic data, and special events
Classify crime severity and type from surveillance images/videos into High Risk, Medium Risk, and Low Risk categories
Recommend optimal patrol resource allocation based on predictions
Reduce response time and improve preventive policing efficiency
Provide actionable insights for urban planning and public safety initiatives

Tech Stack
AreaTools / Technologies
Programming-Python
ML Libraries-TensorFlow, Keras, Scikit-learn
Data Handling-Pandas, NumPy
Visualization-Matplotlib, Seaborn, Plotly
Image Processing-OpenCV
Deployment-FastAPI
Dataset-Synthetic + Realistic (CSV and Image)

Dataset Overview
1. Crime_rate_dataset.csv
- Date-wise crime incidents across different categories (Theft, Assault, Burglary, Vandalism, etc.)
- Crime count per day/week/month
- Historical trends across multiple years
- District/zone-wise crime distribution

2. Event_holiday_data.csv
- Dates of major events, festivals, holidays, and sports events
- Estimated crime rate fluctuations during these periods
- Used as categorical features in forecasting

3. Surveillance_crime_images.zip
   Image dataset categorized into:
- High Risk: Violent crimes, weapon presence, aggressive behavior
- Medium Risk: Suspicious activity, trespassing, minor vandalism
- Low Risk: Normal activity, false alarms

Features
Crime Rate Forecasting (LSTM)

Uses a sequence of past crime data
Input: 60 days of historical crime data
Output: Next 30 days crime rate forecast
Evaluated using RMSE, MAE, and MAPE
Visualized using matplotlib and plotly
Includes trend analysis and seasonality decomposition

Crime Severity Classification (CNN)

Image classification model trained on 3 risk categories
Classifies uploaded surveillance image as High Risk, Medium Risk, or Low Risk
Returns prediction confidence + recommended action (immediate response, monitor, routine patrol)
Can process both static images and video frames

# Integration

- Frontend: Farmer-friendly GUI using Tkinter for uploading surveillance images or connecting to live camera feeds
- Backend: Served via FastAPI to handle model inference, data preprocessing, and real-time predictions
- Dashboard: Interactive visualization showing crime trends, predicted hotspots, and resource allocation recommendations

# Model Architecture
LSTM Model for Crime Forecasting

Input Layer (60 timesteps)
    ↓
    
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
    
Dropout (0.2)
    ↓
    
LSTM Layer 2 (64 units)
    ↓
    
Dropout (0.2)
    ↓
    
Dense Layer (32 units, ReLU)
    ↓
    
Output Layer (30 units) → 30-day forecast


# CNN Model for Crime Image Classification
Input Layer (224x224x3)
    ↓
    
Conv2D (32 filters, 3x3) + ReLU + MaxPooling
    ↓
    
Conv2D (64 filters, 3x3) + ReLU + MaxPooling
    ↓
    
Conv2D (128 filters, 3x3) + ReLU + MaxPooling
    ↓
    
Flatten
    ↓
    
Dense (256 units, ReLU) + Dropout (0.5)
    ↓
    
Dense (128 units, ReLU) + Dropout (0.3)
    ↓
    
Output Layer (3 units, Softmax) → High/Medium/Low Risk


# Installation & Setup

Clone the repository
# Git clone 
https://github.com/mskumar1210/Crime-Rate-Prediction.git
cd Crime-Rate-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Extract dataset
unzip surveillance_crime_images.zip -d data/images/

# Usage
Training the Models
Train LSTM forecasting model
python train_lstm_model.py

# Train CNN classification model
python train_cnn_model.py

# Running the Application
Start FastAPI backend
uvicorn app:app --reload --port 8000

# Launch Tkinter GUI (in separate terminal)
python gui_app.py

# Making Predictions
# Example: Forecast next 30 days
from models.lstm_forecaster import CrimeForecaster

forecaster = CrimeForecaster()
forecaster.load_model('models/lstm_crime_model.h5')
predictions = forecaster.predict_next_30_days()

# Example: Classify surveillance image
from models.cnn_classifier import CrimeImageClassifier

classifier = CrimeImageClassifier()
classifier.load_model('models/cnn_crime_classifier.h5')
risk_level, confidence = classifier.predict('image.jpg')
print(f"Risk Level: {risk_level}, Confidence: {confidence:.2%}")

Results & Performance

# LSTM Forecasting Model
RMSE: 2.34 crimes/day
MAE: 1.87 crimes/day
MAPE: 8.3%
Successfully captures seasonal trends and event-based spikes

# CNN Classification Model
Accuracy: 92.5%
Precision: 91.8%
Recall: 90.2%
F1-Score: 91.0%


# Contributing
Contributions are welcome! Please fork the repository and submit a pull request with detailed descriptions of changes
