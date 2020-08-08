# Classifying-Mammographic-Masses
AIM:

The main aim is to predict whether a mammogram mass is benign or malignant.
Classification based on 6 different attributes - BI-RADS assessment, age, shape, margin, density and severity.

DATASET:

From UCI repository (source: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)

Note: 

Out of all the 6 attributes in the dataset, only 5 will be used in our implementation as BI-RADS is not a "predictive" attribute (it is an assesment of how confident the severity classification is) and so we will discard it.

The dataset needs to be cleaned as there are many NA values, so data preprocessing is a must.
Also, I tried working on different kernel functions of SVM in this project to get a deeper understanding but for that it was needed to normalize the data. 
Before implementing on SVM, the data was normlaized for more accurate and reliable results.

MODELS IMPLEMENTED & ACCURACIES OBTAINED:

1. KNN: 78.55%
2. SVM using:
   -linear kernel: 79.6%
   -radial basis function kernel: 80.1%
   -poly kernel: 79.27%
   -sigmoid kernel: 73.51%
4. Decision Trees: 73.73%
5. Random Forest: 75.28%
6. Logistic Regression: 80.74%
7. Naive Bayes: 78.44%
