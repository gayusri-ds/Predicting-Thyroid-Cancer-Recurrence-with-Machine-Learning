# Thyroid Cancer Recurrence Prediction: A Comparative Analysis of Classification Algorithms

## Overview
This project aims to predict the recurrence of thyroid cancer using various classification algorithms. The dataset was obtained from Kaggle and consists of features related to thyroid cancer patients, including demographic information, medical history, and treatment details.

## Dataset
The dataset used in this project contains information on thyroid cancer patients, including whether they experienced cancer recurrence or not. The features include:
- **Demographic Data**: Age, gender, etc.
- **Medical History**: Previous conditions, treatments, etc.
- **Treatment Details**: Types of treatments received, duration, etc.
The dataset can be accessed [https://www.kaggle.com/datasets/jainaru/thyroid-disease-data]


## Preprocessing
- **Handling Missing Values**: Checked for missing values and applied appropriate methods for handling them, ensuring the integrity of the dataset.
- **Data Encoding**: Categorical features were encoded using label encoding to convert them into numerical format, facilitating model training.
- **Data Scaling**: Numeric features were scaled using `MinMaxScaler` to bring them within a uniform range, preventing bias in model training.
- **Data Splitting**: The dataset was split into training and testing sets to evaluate model performance effectively.

## Models Evaluated
- **Logistic Regression**: Evaluated the baseline performance of logistic regression for thyroid cancer recurrence prediction.
- **Best Logistic Regression**: Utilized `RandomizedSearchCV` to optimize hyperparameters for logistic regression, enhancing model accuracy.
- **Artificial Neural Network (ANN)**: Implemented an ANN model to explore non-linear relationships and potentially improve prediction accuracy.
- **Decision Tree**: Utilized a decision tree classifier to capture complex decision boundaries within the data.
- **Random Forest**: Employed a random forest classifier to leverage the power of ensemble learning for improved prediction.
- **Ensemble Model**: Combined multiple models using a voting classifier to benefit from diverse predictions and enhance overall accuracy.

## Model Evaluation
The performance of each model was evaluated based on both training and testing accuracy. Additionally, classification reports and confusion matrices were generated to assess the precision, recall, and F1-score for each class.

## Results
The results of the comparative analysis are as follows:

| Algorithm                          | Train Accuracy | Test Accuracy |
|------------------------------------|----------------|---------------|
| Logistic Regression                | 0.910653       | 0.876712      |
| Best Logistic Regression           | 0.920962       | 0.890411      |
| ANN                                | 0.993127       | 0.986301      |
| Decision Tree                      | 1.000000       | 0.931507      |
| Best Decision Tree                 | 0.965636       | 0.958904      |
| Random Forest                      | 1.000000       | 0.958904      |
| Best Random Forest                 | 0.969072       | 0.931507      |
| Ensemble Model                     | 1.000000       | 0.931507      |
| Best Ensemble Model                | 0.969072       | 0.958904      |

## Conclusion
This project demonstrates the effectiveness of various classification algorithms in predicting thyroid cancer recurrence. While all models achieved high accuracy, the ANN model showed the best performance on both training and testing datasets. The findings provide valuable insights into the application of machine learning techniques in healthcare for predicting cancer outcomes.

## Future Work
Future work may involve:
- Exploring additional features.
- Fine-tuning model hyperparameters further.
- Evaluating the models on larger datasets for robustness and generalization.

## Dependencies
The project utilizes the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `sweetviz`
- `joblib`
- `scipy`
- `sklearn`
- `tensorflow`

## Usage
To run the project:
1. Ensure all required libraries are installed.
2. Download the dataset from Kaggle and place it in the project directory.
3. Execute the provided Python script to preprocess the data, conduct exploratory data analysis, and train different models for prediction.

## Files and Outputs
- `diagnosed_cbc_data_v4.csv`: Input dataset containing thyroid cancer patient data.
- `Thyroid_cancer_eda_report.html`: Automated exploratory data analysis report generated using Sweetviz.
- `log_reg.pkl`, `best_log_reg.pkl`: Saved logistic regression models.
- `trained_models/`: Directory containing trained models for each classification algorithm.

By following these steps, you can replicate the analysis and evaluate the performance of various classification algorithms for thyroid cancer recurrence prediction.
