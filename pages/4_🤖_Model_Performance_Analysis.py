import streamlit as st
import pandas as pd


def main():


    # Set the title and add a horizontal line
    st.title('Model Performance Analysis')
    st.markdown('***')

    # Display model evaluation on train data
    st.write('#### Model evaluation on train data:')
    st.write('##### Confusion matrix:')
    # Display the confusion matrix image
    st.image('images/cm_train.png')

    st.write('##### Classification report:')
    # Read in the training metrics CSV file
    train_metrics = pd.read_csv('data/train_metrics.csv')
    print(train_metrics)
    # Display the DataFrame as a table
    st.dataframe(train_metrics)

    # Add a horizontal line
    st.markdown('***')

    # Display model evaluation on test data
    st.write('#### Model evaluation on test data:')
    st.write('##### Confusion matrix:')
    # Display the confusion matrix image
    st.image('images/cm_test.png')

    st.write('##### Classification report:')
    # Read in the test metrics CSV file
    test_metrics = pd.read_csv('data/test_metrics.csv')
    # Display the DataFrame as a table
    st.dataframe(test_metrics)



if __name__ == '__main__':
    main()
