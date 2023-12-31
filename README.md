# Customer-Churn-Prediction
## Introduction
Predicting customer churn is critical for telecommunication companies to be able to effectively retain customers. It is more costly to acquire new customers than to retain existing ones. For this reason, large telecommunications corporations are seeking to develop models to predict which customers are more likely to change and take actions accordingly.

In this project, we build a model to predict how likely a customer will churn by analyzing its characteristics:

(1) demographic information

(2) account information

(3) services information.

The objective is to obtain a data-driven solution that will allow us to reduce churn rates and, as a consequence, to increase customer satisfaction and corporation revenue.

## Data set
The dataset used in this project is from a fictional telecommunications company and is focused on understanding the relationship between customer characteristics and churn. Here's a summary of the dataset based on the provided information:

### Demographic Information:
1. **gender:** Indicates whether the client is female or male (Female, Male).
2. **SeniorCitizen:** Indicates whether the client is a senior citizen or not (0, 1).
3. **Partner:** Indicates whether the client has a partner or not (Yes, No).
4. **Dependents:** Indicates whether the client has dependents or not (Yes, No).

### Customer Account Information:
5. **tenure:** Number of months the customer has stayed with the company (Multiple different numeric values).
6. **Contract:** Indicates the customer’s current contract type (Month-to-Month, One year, Two years).
7. **PaperlessBilling:** Indicates whether the client has paperless billing or not (Yes, No).
8. **PaymentMethod:** The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit Card (automatic)).
9. **MonthlyCharges:** The amount charged to the customer monthly (Multiple different numeric values).
10. **TotalCharges:** The total amount charged to the customer (Multiple different numeric values).

### Services Information:
11. **PhoneService:** Indicates whether the client has a phone service or not (Yes, No).
12. **MultipleLines:** Indicates whether the client has multiple lines or not (No phone service, No, Yes).
13. **InternetService:** Indicates whether the client is subscribed to Internet service with the company (DSL, Fiber optic, No).
14. **OnlineSecurity:** Indicates whether the client has online security or not (No internet service, No, Yes).
15. **OnlineBackup:** Indicates whether the client has online backup or not (No internet service, No, Yes).
16. **DeviceProtection:** Indicates whether the client has device protection or not (No internet service, No, Yes).
17. **TechSupport:** Indicates whether the client has tech support or not (No internet service, No, Yes).
18. **StreamingTV:** Indicates whether the client has streaming TV or not (No internet service, No, Yes).
19. **StreamingMovies:** Indicates whether the client has streaming movies or not (No internet service, No, Yes).

The target variable is the **Churn** column, which indicates whether the customer departed within the last month or not (Yes, No). The analysis aims to explore how these demographic, account, and services features are related to customer churn. This dataset can be used to build predictive models to identify factors influencing churn and develop strategies for customer retention.

## Exploratory Data Analysis and Data Cleaning
Exploratory data analysis consists of analyzing the main characteristics of a data set usually by means of visualization methods and summary statistics. The objective is to understand the data, discover patterns and anomalies, and check assumptions before performing further evaluations.

### Missing values and data types
At the beginning of EDA, we want to know as much information as possible about the data, this is when the pandas.DataFrame.info method comes in handy. This method prints a concise summary of the data frame, including the column names and their data types, the number of non-null values, and the amount of memory used by the data frame.

## Feature Engineering
Feature engineering is the process of extracting features from the data and transforming them into a format that is suitable for the machine learning model. In this project, we need to transform both numerical and categorical variables. Most machine learning algorithms require numerical values; therefore, all categorical attributes available in the dataset should be encoded into numerical labels before training the model. In addition, we need to transform numeric columns into a common scale. This will prevent that the columns with large values dominate the learning process. The techniques implemented in this project are described in more detail below. All transformations are implemented using only Pandas; however, we also provide an alternative implementation using Scikit-Learn. As you can see, there are multiple ways to solve the same problem.

No modification
The SeniorCitizen column is already a binary column and should not be modified.

## Assessing multiple algorithms
Algorithm selection is a key challenge in any machine learning project since there is not an algorithm that is the best across all projects. Generally, we need to evaluate a set of potential candidates and select for further evaluation those that provide better performance.

In this project, we compare 6 different algorithms, all of them already implemented in Scikit-Learn.

Dummy classifier (baseline) K Nearest Neighbours Logistic Regression Support Vector Machines Random Forest Gradiente Boosting.

As in this project, all models outperform the dummy classifier model in terms of prediction accuracy. Therefore, we can affirm that machine learning is applicable to our problem because we observe an improvement over the baseline.

It is important to bear in mind that we have trained all the algorithms using the default hyperparameters. The accuracy of many machine learning algorithms is highly sensitive to the hyperparameters chosen for training the model. A more in-depth analysis will include an evaluation of a wider range of hyperparameters (not only default values) before choosing a model (or models) for hyperparameter tuning. Nonetheless, this is out of the scope of this article. In this example, we will only further evaluate the model that presents higher accuracy using the default hyperparameters. As shown above, this corresponds to the gradient boosting model which shows an accuracy of nearly 80%.

## Algorithm selected: Gradient Boosting
Gradient Boosting is a very popular machine learning ensemble method based on a sequential training of multiple models to make predictions. In Gradient Boosting, first, you make a model using a random sample of your original data. After fitting the model, you make predictions and compute the residuals of your model. The residuals are the difference between the actual values and the predictions of the model. Then, you train a new tree based on the residuals of the previous tree, calculating again the residuals of this new model. We repeat this process until we reach a threshold (residual close to 0), meaning there is a very low difference between the actual and predicted values. Finally, you take a sum of all model forecasts (prediction of the data and predictions of the error) to make a final prediction.

We can easily build a gradient boosting classifier with Scikit-Learn using the GradientBoostingClassifier class from the sklearn.ensemble module. After creating the model, we need to train it (using the .fit method) and test its performance by comparing the predictions (.predict method) with the actual class values, as you can see in the code above.

the GradientBoostingClassifier has multiple hyperparameters; some of them are listed below:

learning_rate: the contribution of each tree to the final prediction.

n_estimators: the number of decision trees to perform (boosting stages).

max_depth: the maximum depth of the individual regression estimators.

max_features: the number of features to consider when looking for the best split.

min_samples_split: the minimum number of samples required to split an internal node.

The next step consists of finding the combination of hyperparameters that leads to the best classification of our data. This process is called hyperparameter tuning.

## Hyperparameter tuning
The process of hyperparameter tuning is crucial in machine learning for optimizing model performance. It involves testing various combinations of hyperparameters, selecting the best-performing ones based on a chosen metric and validation method. The common practice is to use k-fold cross-validation, splitting the training data into multiple samples for testing and training.

There are different techniques for hyperparameter tuning:

1. **Grid Search:** Tests all combinations in a predefined grid, but can be computationally expensive.

2. **Random Search:** Randomly samples combinations from a specified grid, providing computational efficiency but may not evenly cover the entire grid.

3. **Bayesian Optimization:** An advanced technique that intelligently selects hyperparameters based on past evaluations.

In scikit-learn, random search can be implemented using the `RandomizedSearchCV` class. This involves specifying a grid of hyperparameter values as a dictionary and randomly sampling combinations from this grid. The `n_iter` parameter determines the number of combinations to sample.

After fitting the `RandomizedSearchCV` object, the best hyperparameters can be obtained using the `best_params_` attribute. This information is essential for optimizing the model's performance without exhaustively testing all possible combinations. In a provided example, the best hyperparameters were {‘n_estimators’: 90, ‘min_samples_split’: 3, ‘max_features’: ‘log2’, ‘max_depth’: 3}, indicating the optimal settings for a specific model.

## Performace of the model
In the final phase of the machine learning process, the model's performance is evaluated using the best hyperparameters through the confusion matrix and various evaluation metrics. The confusion matrix provides a detailed breakdown of correct and incorrect classifications, distinguishing between true positives, true negatives, false positives, and false negatives.

Key evaluation metrics derived from the confusion matrix include accuracy, sensitivity (recall), specificity, and precision. Accuracy represents overall correct predictions, sensitivity measures the ability to correctly identify positive instances, specificity gauges the accuracy in identifying negative instances, and precision assesses the accuracy of positive predictions.

Scikit-Learn's `classification_report` function streamlines the presentation of these metrics, providing a comprehensive summary for each class. The analysis of sensitivity and specificity, in this example, reveals insights into the model's ability to predict positive and negative instances.

Overall, the summary emphasizes the importance of considering a range of metrics, especially in imbalanced datasets, to gain a nuanced understanding of the model's strengths and areas for improvement. The example underscores the model's bias toward accurately predicting non-churning customers, a common trait in gradient boosting classifiers.

## Drawing conclusions — Summary
In this post, we have walked through a complete end-to-end machine learning project using the Telco customer Churn dataset. We started by cleaning the data and analyzing. Then, to be able to build a machine learning model, we transformed the categorical data into numeric variables (feature engineering). After transforming the data, we tried 6 different machine learning algorithms using default parameters. Finally, we tuned the hyperparameters of the Gradient Boosting Classifier (best performance model) for model optimization, obtaining an accuracy of nearly 80% (close to 6% higher than the baseline).
