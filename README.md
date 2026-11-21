
# Early Human Intention Prediction Using Machine Learning

## Overview
A machine learning project focused on predicting human activities from temporal motion data using Random Forest classification.

## Technologies Used
- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

## Dataset
LARA (Logistic Activity Recognition Challenge) dataset.

## Project Features
- Human activity classification
- Early intention prediction using partial motion sequences
- Random Forest model training
- Performance evaluation using confusion matrices

## Project Structure
The `HumanIntentionPrediction` folder contains the training scripts, prediction demo, dataset processing code, and evaluation plots.

## Current Implementation

The project currently implements a machine learning-based approach for predicting human activities using temporal motion data from the LARA dataset.

- Trains a Random Forest classifier to recognize human activities from motion features.
- Performs early activity prediction using the initial portion of a motion sequence rather than waiting for the complete activity.
- Evaluates model performance through accuracy scores, classification reports, and confusion matrices.
- Supports prediction using a trained machine learning model.

## Future Development

The long-term objective is to transform the existing offline prediction model into a real-time human intention recognition system.

Planned developments include:

- **Real-Time Recognition:** Integrating live camera-based human pose estimation to recognize ongoing activities.
- **Deep Learning:** Exploring LSTM, GRU, or Transformer-based architectures to better capture temporal motion patterns.
- **Robotics Integration:** Connecting the prediction system with ROS 2 to support human-aware robotic decision-making.
- **Predictive Human-Robot Interaction:** Enabling robots to anticipate human actions and respond proactively.
- **Practical Applications:** Exploring applications in collaborative robotics, assistive robots, industrial safety, and intelligent automation.

## Project Status

**Ongoing Development** — The current implementation demonstrates early human activity prediction using prerecorded motion data. Future work will focus on real-time inference, improved prediction capabilities, and integration with robotic systems.
