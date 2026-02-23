Pima Diabetes Prediction using Neural Networks

This project builds and evaluates a Neural Network model to predict the likelihood of diabetes using the Pima Indians Diabetes Dataset. The implementation is provided in a Jupyter Notebook: pima_diabetes_nn.ipynb

 Project Overview:

The goal of this project is to: Perform data preprocessing and cleaning
Explore and analyze the dataset,Build a Neural Network model,Train and evaluate the model,Measure performance using classification metrics

The model predicts whether a patient is diabetic based on medical attributes.


 Dataset Description

The dataset contains medical diagnostic measurements for female patients of Pima Indian heritage.

Features:Pregnancies,Glucose,Blood Pressure,Skin Thickness,Insulin,BMI (Body Mass Index),Diabetes Pedigree Function,Age

Target:

Outcome

0 = Non-diabetic

1 = Diabetic

 Technologies Used

Python,Jupyter Notebook,NumPy,Pandas,Matplotlib / Seaborn (for visualization),Scikit-learn,TensorFlow / Keras (for Neural Network model)

 Data Preprocessing

The notebook includes:Handling missing or zero values,Feature scaling / normalization,Train-test split

Data visualization and exploration

 Model Architecture

The Neural Network typically includes:

Input layer (based on number of features)

One or more hidden layers with activation functions (e.g., ReLU)

Output layer with Sigmoid activation (for binary classification)

Loss Function: Binary Crossentropy,Optimizer:Adam (or similar)
Evaluation Metrics:Accuracy,Confusion Matrix,Precision,Recall,F1-score

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