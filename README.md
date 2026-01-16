# End-to-end-chest-cancer-problem



---

## Table of contents
- [Summary](#summary)
- [Problem Statement](#problem-statement)
- [Demo](#demo)
- [Dataset](#dataset)
- [Key features](#key-features)
- [Technology Stack](#technology-stack)
- [Quick start](#quick-start)
- [Training](#training)
- [Model Evaluation & MLflow Configuration](#model-evaluation--mlflow-configuration)
- [Results](#results)

---


## Summary
This project is an end-to-end deep learning application for chest cancer classification using medical imaging data. 
The system includes data ingestion,model training, evaluation, a web-based user interface, and cloud deployment on AWS with an automated CI/CD pipeline.
Using VGG16 transfer learning with fine-tuning, the model achieved **94.6% test accuracy**, ranking it **top 1 on the benchmark dataset** for this task.

## Problem Statement
Early detection of chest cancer is critical for improving patient outcomes. 
Manual diagnosis from medical images is time-consuming and subject to human error. 
This project aims to assist medical professionals by automatically classifying chest cancer from medical images using deep learning.

## Demo

## Dataset
- Source: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/code?datasetId=839140&sortBy=dateRun&tab=profile&excludeNonAccessedDatasources=false
- Image Type: CT scan
- Classes: Large Cell Carcinoma / Squamous Cell Carcinoma / Adenocarcinoma / Normal

## Key features
- End-to-end deep learning pipeline for chest cancer classification
- Reproducible data and model pipelines
- Systematic experiment tracking and model lifecycle management
- Web-based interface for real-time model inference
- Automated CI/CD pipeline for testing, building, and deployment
- Containerized and cloud-deployed application
- Modular, scalable, and maintainable project structure

## Technology Stack
- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow
- **MLOps Tools:** DVC, MLflow, Docker, GitHub Actions
- **Containerization:** Docker
- **CI/CD:** GitHub Actions (AWS deployment)
- **Cloud Platform:** AWS (EC2 / ECR)
- **Web Framework:** Flask
- **Version Control:** Git & GitHub

## Quick start
### 1. Clone
```bash
git clone https://github.com/Nikhat-Suleimanov9/end-to-end-chest-cancer-problem.git
cd end-to-end-chest-cancer-problem
```

### 2. Setup environment with Anaconda
```bash
conda create --name <env_name> python=3.9
conda activate <env_name>

pip install -r requirements.txt
```
### 3. Run full DVC pipeline(optional)
```bash
dvc repro
```
### 4. Run web app locally
```bash
python app.py
```
Optionally, you can also run with Docker


## Training
- **Base model**: pre-trained VGG16 
- **Transfer learning approach**:
  1. Added custom fully connected layers on top of VGG16
  2. Initially froze the pre-trained VGG16 layers and trained only the custom layers
  3. Unfroze the last 4 layers of VGG16 and fine-tuned them along with custom layers
- **Training details**:
  - Optimizer: Adam
  - Loss function: Categorical Cross-Entropy
  - Metrics: Accuracy 

## Model Evaluation & MLflow Configuration
Follow the steps below to enable MLflow logging for model evaluation.

---

### Step 1: Enable MLflow Logging

Navigate to the following file: src/cnnChestCancer/pipeline/fourth_stage_model_evaluation.py
Uncomment the MLflow logging line:

```python
evaluation.log_into_mlflow()
```
### Step 2: Enable MLflow Logging Set MLflow Tracking URI:

Navigate to the following file: src/cnnChestCancer/config/configuration.py
Update the mlflow_uri with your MLflow server address:
```python
mlflow_uri = "your_uri"
```

### Results
The model demonstrates strong performance in classifying chest cancer cases, achieving a **test accuracy of 94.6%**.  
Example predictions, along with screenshots of the web application output, are shown below.
(assets/image_1.png)
=======


### Results
The model demonstrates strong performance in classifying chest cancer cases, achieving a **test accuracy of 94.6%**.  
Example predictions, along with screenshots of the web application output, are shown below.

>>>>>>> 654dbacb7a79bc92bcb82204db9d17dcab1b3a4d
