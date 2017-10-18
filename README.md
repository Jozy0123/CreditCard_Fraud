# CreditCard_Fraud
The objective of this analysis is to build a model for predicting potential credit card fraud reasonably and test the results for potential generalization.

This dataset was downloaded from kaggle (link: https://www.kaggle.com/dalpozz/creditcardfraud). In order to protect the privcacy of these credit cards' owners. A PCA was performed by the dataset provider before it was publicly available. However, 2 variables were retained in their original form: Time and Amount (transaction amount).   

The raw dataset consists 30 feature columns and 1 label colunm. It has 284807 rows (examples). Since every example was labeled, it is a supervised learning problem. More specifically, a supervised classfication problem. 

## Data Exploration (Time variable was dropped)
This dataset is a highly imbalanced dataset, since majority labels were 0.
Since PCA (Principle Component Analysis) is generally used for dimension reduction. A simple scatter plot was used for viewing the first principle components, which have the most variance. By only using the first 2 principle components, seperating these 2 classes is not possible. 

A histgram was plotted for the feature - Amount, since it is the only interpretable variable. It is highly skewed as most of the amount spending in a transaction concentrated on small numbers. 

A correlation heatmap was also plotted. Since the variable Amount was not used for PCA, it still has correlation with some principle components. If logistic regression was performed, a regularization parameter could be used to remedy the multicollinearty. 

## Scikit-learn machine learning models
In order to evaluate the performance of a trained model, the original dataset was split into stratified (perserve the ratio of postive and negative examples) training set (70%) and test set (30%) stratifically.  Logictic regression (with regularization) with 10-fold cross-validation was used for fitting the trainining set. It was found that a reasonably higher regularization term could be helpful for preventing overfitting (could be explained by the multicollinearty) based on precision scores (which is the true positive divided by the sum of false postive and true positive). However, when the model was used to classify the test set examples, precision score was only about 0.50. A random forest was also used.

Hyperparameters tuning might be able to increase the overall prediction accuracy. A better strategy is to use oversampling or undersampling. But in order to use all the information from the original set, instead, a 3 layer deep neural network was used. 

## Tensorflow
A three layer deep neural network (hidden units: 10, 20, and 10) was used to fit the trainining set. The fitted model was evaluated via predicating the class of test set examples. The result showed that area under precision and recall curve (good indicator for imbalanced dataset) was 0.82. Tuning with hyperparameters could even further help to improve the performance. 

## Conclusion:
Even without the original information and after dropping one variable, deep neurual network still managed to achieve a reasonable performance. It is much better than Logistic Regression and Random Forest.
