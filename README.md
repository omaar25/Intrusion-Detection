# Intrusion Detection

## Overview
This project focuses on developing an Intrusion Detection System (IDS) using various machine learning techniques. The system employs Exploratory Data Analysis to understand attack techniques and subsequently implements LSTM (Long Short-Term Memory networks), Conv1D (1D Convolutional Neural Networks), XGBoost, and Random Forest algorithms to detect intrusions effectively.
## Dataset
The dataset used in this project is [[edgeiiotset cyber security](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)]. It consists of network traffic data labeled with various attack techniques, including:

- DDoS_UDP
- DDoS_ICMP
- SQL_injection
- DDoS_TCP
- Vulnerability_scanner
- Password
- DDoS_HTTP
- Uploading
- Backdoor
- Port_Scanning
- XSS
- Ransomware
- Fingerprinting
- MITM

## Exploratory Data Analysis (EDA)
The EDA for this project is documented in the `intrusion-detection-eda.ipynb` notebook. In this notebook, we aimed to understand the dataset and analyze various attack techniques. The focus was on exploring the characteristics of the data, identifying patterns, and gaining insights into the different types of attacks present in the dataset.

## Model Performance Comparison
In this project, we implemented various models to detect intrusions and evaluated their performance using metrics such as accuracy, error rate, recall, precision, and F1 score. The details are documented in the `intrusion-detection-models.ipynb` notebook. Here are the summarized results for each model:

### 1. LSTM Model Architecture
The LSTM model is designed to capture temporal dependencies in sequential data, making it suitable for intrusion detection tasks that involve time-series analysis. The architecture consists of the following layers:

- **Input Layer:** Accepts input sequences with a specified shape.
- **LSTM Layer 1:**
  - Units: 256
  - Activation Function: Tanh
  - Return Sequences: True (to feed into the next LSTM layer)
- **LSTM Layer 2:**
  - Units: 128
  - Activation Function: Tanh
- **Output Layer:**
  - Dense Layer: Fully connected layer
  - Units: Number of classes (for classification)
  - Activation Function: Softmax

**Performance Metrics:**
{'Accuracy': 0.94, 'Error': 0.06, 'Recall': 0.93, 'Precision': 0.95, 'F1 Score': 0.94}

### 2. CNN Model Architecture
The CNN model leverages convolutional layers to extract features from the input sequences, providing a powerful mechanism for intrusion detection. The architecture includes the following components:

- **Input Layer:** Accepts input sequences with a specified shape.
- **Convolutional Layer 1:**
  - Filters: 32
  - Kernel Size: 3
  - Padding: Same
  - Activation Function: ReLU
- **Max Pooling Layer 1:**
  - Strides: 2
- **Convolutional Layer 2:**
  - Filters: 64
  - Kernel Size: 3
  - Padding: Same
  - Activation Function: ReLU
- **Max Pooling Layer 2:**
  - Pool Size: 2
  - Strides: 2
  - Padding: Same
- **Convolutional Layer 3:**
  - Filters: 128
  - Kernel Size: 3
  - Activation Function: ReLU
- **Max Pooling Layer 3:**
  - Pool Size: 2
  - Strides: 2
  - Padding: Same
- **Flatten Layer:** Converts the 3D output to 1D.
- **Fully Connected Dense Layers:**
  - Dense Layer 1: 64 units, ReLU activation
  - Dense Layer 2: 32 units, ReLU activation
- **Output Layer:**
  - Dense Layer: Fully connected layer
  - Units: Number of classes (for classification)
  - Activation Function: Softmax

**Performance Metrics:**
{ 'Accuracy': 0.8050, 'Error': 0.1950, 'Recall': 0.8050, 'Precision': 0.8689, 'F1 Score': 0.7993 }


### 3. Random Forest
**Performance Metrics:**
{'Accuracy': 0.95, 'Error': 0.04, 'Recall': 0.95, 'Precision': 0.95, 'F1 Score': 0.95}

### 4. XGBoost
**Performance Metrics:**
{'Accuracy': 0.94, 'Error': 0.06, 'Recall': 0.94, 'Precision': 0.96, 'F1 Score': 0.94}


## Best Model Selection
Among the models evaluated, **XGBoost** demonstrated the highest recall at **0.94**, indicating that it successfully identified a significant proportion of actual positive instances (i.e., intrusions). The performance metrics for each model can be summarized as follows:

- **LSTM:** Recall = 0.7173
- **CNN:** Recall = 0.8050
- **Random Forest:** Recall = 0.8642
- **XGBoost:** Recall = 0.9322

### Why Recall Matters
Recall is a crucial metric in intrusion detection systems for several reasons:

- **False Negatives Impact:** In the context of intrusion detection, a false negative (missing an actual intrusion) can have severe consequences, potentially leading to security breaches. High recall ensures that most of the actual intrusions are detected.

- **Security Sensitivity:** Organizations prioritize detecting intrusions over falsely identifying them (false positives). A model that maximizes recall helps in alerting security personnel of potential threats.

- **Real-World Applications:** In practical applications, high recall reduces the risk of undetected intrusions, which is paramount in cybersecurity settings. A model with higher recall ensures better coverage of potential attacks.

