📞 Telecom Customer Churn Prediction Engine

This project demonstrates a full-stack deployment of a Machine Learning model designed to predict customer churn risk in real-time, enabling proactive retention strategies.

Project Goal

The primary objective was to minimize False Negatives (missed churners) to maximize revenue protection. This was achieved by optimizing for high Recall on the minority class.

🎯 Final Model Performance

Decision Tree Classifier :
Churn Recall (Capture Rate) -> 79%
Churn Precision (Efficiency) -> 49%
Missed Customers (FN) -> 80

Technical Pipeline (MLOps)

The application moves data from the frontend to the model and back in three key steps:

Frontend (UI): Built with HTML/Tailwind CSS and JavaScript for real-time user input and visual risk display.

API Server (MLOps): A Python/Flask server loads the trained Decision Tree ($\mathbf{churn\_model.pkl}$) using Joblib.

Prediction Logic: The server executes the full 18-feature engineering pipeline and predicts the risk score, which is then translated into an URGENT or LOW RISK business action based on a 45% threshold.

Key Data Science Techniques Used

Feature Engineering: Creation of custom metrics like Charge Deviation and Loyalty Flag.

Imbalance Handling: Use of the class_weight='balanced' parameter to boost the model's sensitivity to the rare Churn class.

Deployment Strategy: Deployment of an unscaled Decision Tree model to simplify the API pipeline.
