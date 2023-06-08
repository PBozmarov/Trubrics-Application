import streamlit as st
def intro():
    st.title("🌸 Iris Dataset ML Application")
    st.write("""This is an application that assists homeowners in identifying the types of iris flowers growing in their garden.
                As the ML model that we are using is not perfect, we have incorporated user feedback within the application to help to identify issues.
                The application is made to be be user-friendly and easily understandable by individuals who are not familiar with ML.
            """)
    #st.write('This is an application that assists homeowners in identifying the types of iris flowers growing in their garden.')
    #st.write("As the ML model that we are using is not perfect, we have incorporated user feedback within the application to help to identify issues.")
    #st.write('The application is made to be be user-friendly and easily understandable by individuals who are not familiar with ML.')
    st.markdown("***")
    st.write('The application consists of 4 pages:')
    st.markdown('- Data Insights' + ' : Insights for the dataset in terms of various plots and statistical analysis')
    st.markdown('- Custom Predictions' + ' : We can make predictions on custom inputs')
    st.markdown('- Test Predictions' + ' : Perform predictions on test data')
    st.markdown('- Model Performance Analysis' + ' : Analyze model performance using metrics like accuracy, precision, recall, etc. and various plots')


if __name__ == "__main__":
    intro()