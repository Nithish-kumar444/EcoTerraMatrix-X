# ECOTERRAMATRIXX
## Global Climate Intelligence Matrix
### AI-Powered Carbon Emission Analytics & Forecasting Platform
## Overview
ECOTERRAMATRIXX is an advanced Climate Intelligence & Carbon Analytics platform built using Streamlit and Plotly.

## The system delivers:
 . Global CO₂ emission visualization
 . AI-powered carbon estimation models
 . Renewable energy impact analytics
 . Deep Learning-based pollution classification
 . Forecasting for environmental decision-making

## Key Features
 . Global Emission Dashboard
 . Real-time CO₂ emission visualization
 . Continent-level emission share gauge
 . Interactive world emission heatmap (Choropleth Map)
 . Historical emission trend analysis

## Advanced Climate Analytics
 . Continent-wise emission comparison
 . Renewable Energy vs CO₂ correlation analysis
 . Cumulative CO₂ emissions over time
 . Time-series forecasting visualization

## EcoTerraMatrix-X: Carbon Estimation & Analysis Dashboard
   An intelligent carbon calculator powered by Machine Learning.

## Multi-Sector Carbon Estimation:
 . Individual Carbon Footprint
 . Transport Emissions
 . Industrial Emissions
 . Scope 3 Emissions

## Powered by:
 ### Random Forest Regressor (Scikit-learn)

## Carbon Emission Classifier (Deep Learning)
### Upload an image of:
 . Cityscape
 . Industrial region
 . Sky pollution
 . The CNN model predicts:
 . Pollution Level
 . Environmental Degradation Category

## Powered by:
TensorFlow / Keras CNN Model
Image classification pipeline

## Machine Learning Models
### Random Forest Regressor

## Used for:
### Carbon emission estimation
### Multi-sector prediction models

## Framework:
Scikit-learn

## Convolutional Neural Network (CNN)

## Used for:
### Pollution classification from uploaded images

## Framework:
### TensorFlow / Keras

## Technical Stack

 | Layer         | Technology                              |
| ------------- | --------------------------------------- |
| Frontend      | Streamlit                               |
| Visualization | Plotly (Gauge, Choropleth, Time-Series) |
| Backend       | Python                                  |
| ML Framework  | Scikit-learn                            |
| DL Framework  | TensorFlow / Keras                      |
| Deployment    | Ngrok (Secure Tunnel)                   |

## Installation & Setup
### Clone the Repository
git clone https://github.com/your-username/ECOTERRAMATRIXX.git
cd ECOTERRAMATRIXX

### Install Dependencies
Ensure Python 3.9+ is installed.
pip install -r requirements.txt
## or
pip install streamlit pyngrok pandas numpy scikit-learn tensorflow plotly

## Project Structure

ECOTERRAMATRIXX/
│
├── app.py
├── model/
│   ├── ml_model.pkl
│   ├── dl_model.pkl
│   ├── carbon_emission_model.keras
│   ├── class_names.pkl
│
├── requirements.txt
└── README.md

## Machine Learning Model Files

Download and place inside the model/ directory:

### ML Models (.pkl files)

https://drive.google.com/file/d/1_fZ__caRCFRakam0fR0-flzxSYwzJMaD/view?usp=drive_link
https://drive.google.com/file/d/13Aywt-fM8K6n8AdKGlOWCbW89W4KvJw6/view?usp=drive_link
https://drive.google.com/file/d/1n1dxtSM0ZcVmb565QWjGSaOMzM_89-5c/view?usp=drive_link
https://drive.google.com/file/d/10pyik0KkI0e9PVtW19FkmnPSgJkwBJfi/view?usp=drive_link
https://drive.google.com/file/d/1v-8__qYcEZocPAZyepPBTsYJiEZdNru7/view?usp=drive_link
https://drive.google.com/file/d/1dpfwxdndv2pfxMNCOr34qIcEgVfBwo5i/view?usp=drive_link
https://drive.google.com/file/d/1SO_G0uJllqELzjstuCHA9pnU_XAWZwKu/view?usp=drive_link
https://drive.google.com/file/d/1EevUJVqzccOLOrysFKcraYRs3z5i7w3n/view?usp=drive_link

## DL Model Files
carbon_emission_model.keras
class_names.pkl

## Running the Application
streamlit run app.py
### The app will launch at:
http://localhost:8501

## Remote Deployment (Ngrok)
### If running in:
Google Colab
Jupyter Notebook
Remote server

### Use:
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_AUTH_TOKEN")
public_url = ngrok.connect(8501)
print(public_url)
### This generates a secure public URL.

## Author
### Nithish Kumar, Om sri Abhiram A, Nikheel
