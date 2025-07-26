
# Bank Customer Churn Prediction using Deep Learning

This project focuses on predicting customer churn in a bank using deep learning techniques. Churn prediction helps businesses identify customers who are likely to leave, enabling proactive retention strategies.

## 🔍 Overview

The goal of this project is to build a neural network model that can classify whether a customer is likely to churn based on features like credit score, age, tenure, balance, number of products, and more.

## 📁 Dataset

The dataset used in this project is `Churn_Modelling.csv`, which contains customer data such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Exited (Target variable)

## 🛠️ Tools & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib / Seaborn

## 🧠 Deep Learning Model

The model is built using Keras Sequential API and includes:

- Input Layer (with preprocessing and encoding)
- Dense hidden layers with ReLU activation
- Output Layer with Sigmoid activation for binary classification
- Optimizer: Adam
- Loss: Binary Crossentropy
- Evaluation Metrics: Accuracy, Confusion Matrix

## 📊 Evaluation

Model performance is evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report

## 📌 How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/rahulpoojith/Bank-Customer-Churn-using-deepl-learning.git
    cd Bank-Customer-Churn-using-deepl-learning
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook Bank-Customer-Churn.ipynb
    ```

## 📈 Results

The deep learning model achieved good performance in predicting churn, showcasing the importance of using artificial neural networks for classification tasks in customer behavior analysis.

## 🧑‍💻 Author

- **Rahul Poojith**  
  [GitHub Profile](https://github.com/rahulpoojith)

## 📄 License

This project is licensed under the MIT License.
