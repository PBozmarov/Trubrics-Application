
import streamlit as st
import pandas as pd

def main():
    df = pd.read_csv('data/df_all.csv')
    st.title("Iris Data Insights")
    st.markdown("***")
    
    # Display dataset summary
    st.markdown("""The Iris dataset comprises measurements of four distinct features collected from three different species of iris flowers.
                   Each sample within the dataset represents an individual iris flower, and the objective is to predict the species of the 
                   flower based on its measured features.""")

    # Display key dataset information
    st.subheader("Dataset Information:")
    st.write("- Dataset Size: 150 samples")
    st.write("- Feature Count: 4 (sepal length, sepal width, petal length, petal width)")
    st.write("- Class Count: 3 (setosa, versicolor, virginica)")
    st.write("- Class Distribution: Balanced dataset with 50 samples per species")
    st.write("- Data Type: Numerical values representing continuous measurements")
    st.write()

    # Images of the Iris flowers
    SPECIES_IMAGE_DICT = {
        0: "images/iris_setosa.png",
        1: "images/iris_versicolor.png",
        2: "images/iris_virginica.png",
    }

    st.markdown("***")
    st.subheader("Images of the Iris Flowers:")
    # Display the images in 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(SPECIES_IMAGE_DICT[0])
    with col2:
        st.image(SPECIES_IMAGE_DICT[1])
    with col3:
        st.image(SPECIES_IMAGE_DICT[2])

    st.markdown("***")

    st.subheader("Data:")

    # Load the data
    df = pd.read_csv('data/df_all.csv')
    st.dataframe(df)
    st.markdown("***")

    # Display the summary statistics
    st.subheader("Summary Statistics:")
    st.image('images/data_insights/stats.png')
    st.markdown("***")

    # Display the correlation matrix
    st.subheader("Correlation Matrix:")
    st.image('images/data_insights/correlation.png')
    st.markdown("***")

    # Display pair plots and distributions
    st.subheader("Pair Plots and Distributions:")
    st.image('images/data_insights/plots.png')
    st.markdown("***")
    

if __name__ == '__main__':
    main()
