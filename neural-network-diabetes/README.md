Pima Diabetes Prediction using Neural Networks

Diabetes Prediction using Neural Networks
Problem Statement

Diabetes is a chronic  disease that requires early detection to prevent long term complications. The objective of this project is to build a machine learning model capable of predicting whether a patient has diabetes based on clinical diagnostic measurements.

The model is trained using the Pima Indians Diabetes Dataset, which includes medical features such as glucose level, blood pressure, BMI, insulin level, age, and number of pregnancies. The target variable (Outcome) indicates whether a patient is diabetic (1) or not (0). The goal is to develop a predictive system that can support early diabetes screening using these features.

Approach

This project follows a complete machine learning workflow:

1. Data Preprocessing

The dataset was loaded and split into training (80%) and testing (20%) sets using stratified sampling to preserve class distribution.

Feature scaling was applied using StandardScaler to normalize input features and improve neural network training stability.

Early stopping was implemented to prevent overfitting by monitoring validation loss during training.

2. Model Architecture

A feedforward neural network was built using TensorFlow/Keras with the following structure:

Input layer: 8 medical features

Hidden layer 1: 16 neurons with ReLU activation

Hidden layer 2: 8 neurons with ReLU activation

Output layer: 1 neuron with Sigmoid activation for binary classification

The model was trained using the Adam optimizer with binary crossentropy loss and binary accuracy as the evaluation metric.

3. Evaluation Metrics

Model performance was evaluated on a test set that was not used during training. The following metrics were calculated:

Accuracy,Precision,Recall,F1 Score,Confusion Matrix

Training and validation accuracy/loss curves were also generated to analyze learning behavior and detect overfitting.

Results

The final model achieved the following performance on the test set:

Test Accuracy: 73.38%

Precision: 0.6226

Recall: 0.6111

F1 Score: 0.6168

The results indicate that the model captures meaningful relationships between the clinical features and diabetes outcomes. While overall accuracy is acceptable for a baseline neural network, recall shows that some diabetes cases are not detected, which is an important consideration in medical applications.

All evaluation outputs, including the confusion matrix and training curves, are saved in the results/ directory.

Analysis:

This neural network model is trained to predict diabetes using the Pima Indians Diabetes dataset. After preprocessing, scaling, and training with early stopping, the model achieved a test accuracy of 73.38% and a test loss of 0.5293, This performance is consistent with basic neural network results reported for this dataset.

 The model achieved a precision of 0.6226, meaning that when the model predicts a patient has diabetes, it is correct 62% of the time. The recall was 0.6111, indicating that the model correctly identifies about 61% of actual diabetes cases. The F1 score of 0.6168 reflects a balanced trade-off between precision and recall, suggesting moderate but stable predictive performance.

From a medical perspective, recall is particularly important because false negatives (undiagnosed diabetes cases) can delay treatment and increase health risks. With a recall of 61%, the model misses approximately 39% of true diabetes cases, which indicates room for improvement in clinical sensitivity. While the overall accuracy is acceptable for a baseline model, optimizing for higher recall would make the system more suitable for screening purposes.



 How to Run the Project
1- Clone the Repository
git clone : https://github.com/ranayasser-lab/task1.git
2- Install Required Libraries
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
3️- Run the Notebook
jupyter notebook

Open:

pima_diabetes_nn.ipynb

Run all cells to train and evaluate the model.

 Results

The model performance is evaluated using: Training accuracy,Test accuracy,Confusion matrix,Classification report

Results may vary depending on:Network architecture,Number of epochs,Learning rate,Data preprocessing steps