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

Missing values and data types
At the beginning of EDA, we want to know as much information as possible about the data, this is when the pandas.DataFrame.info method comes in handy. This method prints a concise summary of the data frame, including the column names and their data types, the number of non-null values, and the amount of memory used by the data frame.

## Feature Engineering
Feature engineering is the process of extracting features from the data and transforming them into a format that is suitable for the machine learning model. In this project, we need to transform both numerical and categorical variables. Most machine learning algorithms require numerical values; therefore, all categorical attributes available in the dataset should be encoded into numerical labels before training the model. In addition, we need to transform numeric columns into a common scale. This will prevent that the columns with large values dominate the learning process. The techniques implemented in this project are described in more detail below. All transformations are implemented using only Pandas; however, we also provide an alternative implementation using Scikit-Learn. As you can see, there are multiple ways to solve the same problem.

No modification
The SeniorCitizen column is already a binary column and should not be modified.

## Assessing multiple algorithms
Algorithm selection is a key challenge in any machine learning project since there is not an algorithm that is the best across all projects. Generally, we need to evaluate a set of potential candidates and select for further evaluation those that provide better performance.

In this project, we compare 6 different algorithms, all of them already implemented in Scikit-Learn.

Dummy classifier (baseline) K Nearest Neighbours Logistic Regression Support Vector Machines Random Forest Gradiente Boosting

As shown below, all models outperform the dummy classifier model in terms of prediction accuracy. Therefore, we can affirm that machine learning is applicable to our problem because we observe an improvement over the baseline.

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
Thus far we have split our data into a training set for learning the parameters of the model, and a testing set for evaluating its performance. The next step in the machine learning process is to perform hyperparameter tuning. The selection of hyperparameters consists of testing the performance of the model against different combinations of hyperparameters, selecting those that perform best according to a chosen metric and a validation method.

For hyperparameter tuning, we need to split our training data again into a set for training and a set for testing the hyperparameters (often called validation set). It is a very common practice to use k-fold cross-validation for hyperparameter tuning. The training set is divided again into k equal-sized samples, 1 sample is used for testing and the remaining k-1 samples are used for training the model, repeating the process k times. Then, the k evaluation metrics (in this case the accuracy) are averaged to produce a single estimator.

It is important to stress that the validation set is used for hyperparameter selection and not for evaluating the final performance of our model, as shown in the image below.

There are multiple techniques to find the best hyperparameters for a model. The most popular methods are (1) grid search, (2) random search, and (3) bayesian optimization. Grid search test all combinations of hyperparameters and select the best performing one. It is a really time-consuming method, particularly when the number of hyperparameters and values to try are really high.

In random search, you specify a grid of hyperparameters, and random combinations are selected where each combination of hyperparameters has an equal chance of being sampled. We do not analyze all combinations of hyperparameters, but only random samples of those combinations. This approach is much more computationally efficient than trying all combinations; however, it also has some disadvantages. The main drawback of random search is that not all areas of the grid are evenly covered, especially when the number of combinations selected from the grid is low.

We can implement random search in Scikit-learn using the RandomSearchCV class from the sklearn.model_selection package.

First of all, we specify the grid of hyperparameter values using a dictionary (grid_parameters) where the keys represent the hyperparameters and the values are the set of options we want to evaluate. Then, we define the RandomizedSearchCV object for trying different random combinations from this grid. The number of hyperparameter combinations that are sampled is defined in the n_iter parameter. Naturally, increasing n_iter will lead in most cases to more accurate results, since more combinations are sampled; however, on many occasions, the improvement in performance won’t be significant.

After fitting the grid object, we can obtain the best hyperparameters using best_params_attribute. As you can above, the best hyperparameters are: {‘n_estimators’: 90, ‘min_samples_split’: 3, ‘max_features’: ‘log2’, ‘max_depth’: 3}.

## Performace of the model
The last step of the machine learning process is to check the performance of the model (best hyperparameters ) by using the confusion matrix and some evaluation metrics.

Confusion matrix The confusion matrix, also known as the error matrix, is used to evaluate the performance of a machine learning model by examining the number of observations that are correctly and incorrectly classified. Each column of the matrix contains the predicted classes while each row represents the actual classes or vice versa. In a perfect classification, the confusion matrix will be all zeros except for the diagonal. All the elements out of the main diagonal represent misclassifications. It is important to bear in mind that the confusion matrix allows us to observe patterns of misclassification (which classes and to which extend they were incorrectly classified).

In binary classification problems, the confusion matrix is a 2-by-2 matrix composed of 4 elements:

TP (True Positive): number of patients with spine problems that are correctly classified as sick. TN (True Negative): number of patients without pathologies who are correctly classified as healthy. FP (False Positive): number of healthy patients that are wrongly classified as sick. FN (False Negative): number of patients with spine diseases that are misclassified as healthy.

Now that the model is trained, it is time to evaluate its performance using the testing set. First, we use the previous model (gradient boosting classifier with best hyperparameters) to predict the class labels of the testing data (with the predict method). Then, we construct the confusion matrix using the confusion_matrix function from the sklearn.metrics package to check which observations were properly classified. The output is a NumPy array where the rows represent the true values and the columns the predicted classes.

As shown above, 1402 observations of the testing data were correctly classified by the model (1154 true negatives and 248 true positives). On the contrary, we can observe 356 misclassifications (156 false positives and 200 false negatives).

Evaluation metrics Evaluating the quality of the model is a fundamental part of the machine learning process. The most used performance evaluation metrics are calculated based on the elements of the confusion matrix.

Accuracy: It represents the proportion of predictions that were correctly classified. Accuracy is the most commonly used evaluation metric; however, it is important to bear in mind that accuracy can be misleading when working with imbalanced datasets. Sensitivity: It represents the proportion of positive samples (diseased patients) that are identified as such. Specificity: It represents the proportion of negative samples (healthy patients) that are identified as such. Precision: It represents the proportion of positive predictions that are actually correct.

We can calculate the evaluation metrics manually using the numbers of the confusion matrix. Alternatively, Scikit-learn has already implemented the function classification_report that provides a summary of the key evaluation metrics. The classification report contains the precision, sensitivity, f1-score, and support (number of samples) achieved for each class.

As shown above, we obtain a sensitivity of 0.55 (248/(200+248)) and a specificity of 0.88 (1154/(1154+156)). The model obtained predicts more accurately customers that do not churn. This should not surprise us at all, since gradient boosting classifiers are usually biased toward the classes with more observations.

As you may have noticed, the previous summary does not contain the accuracy of the classification. However, this can be easily calculated using the function accuracy_score from the metrics module.

## Drawing conclusions — Summary
In this post, we have walked through a complete end-to-end machine learning project using the Telco customer Churn dataset. We started by cleaning the data and analyzing. Then, to be able to build a machine learning model, we transformed the categorical data into numeric variables (feature engineering). After transforming the data, we tried 6 different machine learning algorithms using default parameters. Finally, we tuned the hyperparameters of the Gradient Boosting Classifier (best performance model) for model optimization, obtaining an accuracy of nearly 80% (close to 6% higher than the baseline).
