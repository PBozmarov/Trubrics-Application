# AI Engineer Intern - Technical Test

## Objective

An application that assists homeowners in identifying the types of [iris](https://www.kaggle.com/datasets/uciml/iris) flowers growing in their garden. As the ML model that we'll be using is not perfect, we have incorporated user feedback within the application to help to identify issues. We have kept in mind that the application should be user-friendly and easily understandable by individuals who are not familiar with ML.

## Install the requirements

```
pip install -r requirements.txt
```

## Run the application
We have created a multi-page application that can be run using the following command:
```
streamlit run home.py
```
## Home Page
The home page is the first page that the user will see. It contains a brief description of the application and a button to navigate to the prediction page.

## Custom Predictions Page
The custom predictions page is where the user can input their own measurements to predict the type of iris flower. The user can input the measurements in using the given sliders. The user can also click on the feedback button to give feedback on the prediction. The feedback is stored in a csv file in the local filesystem.

## Test Predictions Page
The test predictions page is where the user can see the predictions for the test data. The user can also click on the feedback button to give feedback on the prediction. The feedback is stored in a csv file in the local filesystem.

## Model Performance Page
The model performance page is where the user can see the performance of the model expressed with confusion matrix and classification report for the test data and the training data. 


