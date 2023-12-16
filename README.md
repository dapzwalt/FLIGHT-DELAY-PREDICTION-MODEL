# FLIGHT-DELAY-PREDICTION-SUPERVISED-MACHINE LEARNING-MODEL

![image](https://github.com/dapzwalt/FLIGHT-DELAY-PREDICTION-MODEL/assets/125368548/cf786456-9034-4ce3-91ac-85edfe173203)

## Table of Contents
- [Project Overview](#project-overview)
- [Project Objective](#project-objective)
- [Data Sources](#data-sources)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Model](#machine-learning-model)
- [Evaluation Metrics](#evaluation-metrics)
- [Model Optimization](#model-optimization)
- [Key Insights](#key-insights)
- [Challenges](#challenges)
- [Conclusion](#conclusion)

## Project Overview
Flight delays lead to inconvenience for passengers, financial losses for airlines, and operational challenges for airports. Existing prediction models often fall short in accuracy due to the complex interplay of multiple variables. This project addresses these limitations by employing 
advanced machine learning techniques to create a more reliable and precise flight delay prediction system.

## Project Objective
The aim of this project is to build advanced machine learning models that analyze historical flight data and relevant factors (e.g., weather conditions, air traffic) to predict flight delays with a high level of accuracy. Assist airlines in optimizing their operations by providing early insights into
potential delays. This includes optimizing crew schedules, resource allocation, and maintenance planning to minimize disruptions.

## Data Sources
The dataset used in this project was provided by 10Alytics. The dataset contains a collection of features extracted from the company database which includes features as 
Year, Airline, Flight Number, Tail Number, Departure Delay, Arrival delay, Scheduled time, distance and other relevant information

## Exploratory Data Analysis
I performed data wrangling, conducted univariate, bivariate analysis, and multivariate analysis showing correlations between variables and drawing out inferences. checked for missing values and duplicates in my dataset. 
Most importantly, feature engineering was performed to extract relevant information from my raw data and dropping irrelevant variables.

## Data Preprocessing
I performed exploratory data analysis, encoding of categorical variables, normalization of my data which entails scaling the features using Standard Scaler before feeding the data into the machine learning model.

## Machine Learning Model
The Flight delay prediction model is built using a supervised machine learning approach. Training and testing of my data was done by splitting my data to the ratio 70:30. Several classification algorithms were employed which includes:

- **Logistic Regression**
- **Naive Bayes**

## Evaluation Metrics
The following evaluaton metrics were used to assess the performance of the individual machine learning algorithms, which are: 

- **Precision:** This shows that when it predicts that flight has not been delayed in some airports
- **Recall:** This shows the flight delay true positive rate.
- **F1score:** This is the harmonic mean of precision and recall, thus providing a balanced metric for model evaluation
- **Accuracy:** Accuracy shows how good the model performed but is not a final resort to determining or choosing the best model
  
## Model Optimization
After extensive experimentation and hyperparameter tuning, the model with the highest score in terms of its classification report was chosen based on its performance and generalized capabilities.

## Key insights
- The purpose of this project is to predict flight delay with a high level of accuracy.
- After subjecting the models to evaluation metrics like the accuracy, recall, precision and f1score, the best performing model was chosen which is the SVM because we had lesser error with this model
- From the confusion matrix, True positives, a total of 958 customers did not churn meaning they didn't cancel their service and the algorithm predicted right. False negatives a total of 198 customers did not churn meaning that they didnt cancel their service and the algorithm predicted that they churn. So I would advice the company to be more concerned about the false negatives. 

## Challenges
The challenge I faced in this project was in cleaning the data specifically handling the missing values. I needed to use two target variables to experiment my model and as such used variables that would give me a good model prediction when using arrival 
delay as my target variable to train the model and drop some variables and also use some variables i dropped earlier when using departure delay as my target variable. Also selecting the right features or variables that will give a good prediction rather than choosing irrelevant or redundant features can lead to suboptimal model performance.

## Conclusion
Flight delay uses machine learning model to predict whether flight will be delayed or not and which airlines experience more delays than the other to make customers make better decisions. In this project, we are able to see how various features contributes to the flight delay, looking through the jupyter notebook posted will explain more better as I made it step by step for your understanding. However, if there are adjustment to be made or corrections, kindly drop it in. 
Additionally, for production at the backend users like the software developers, an output for software has been made also. Enjoy 
