import streamlit as st
import pandas as pd
from joblib import load
from trubrics.integrations.streamlit import FeedbackCollector
import os

def feedback_example(feedback_type, collector, metadata, open_feedback_label=None, user_response=None):
    feedback_dir = 'feedback'  # Name of the feedback folder
    os.makedirs(feedback_dir, exist_ok=True)  # Create the feedback folder if it doesn't exist

    file_name = os.path.join(feedback_dir, f"{feedback_type}_feedback.json")
    
    if user_response:
        feedback = collector.st_feedback(
            feedback_type=feedback_type, metadata=metadata, path=file_name, user_response=user_response
        )
    else:
        code_snippet = f"""
        from trubrics.integrations.streamlit import FeedbackCollector
        collector = FeedbackCollector()
        collector.st_feedback(
            feedback_type="{feedback_type}"{f', open_feedback="{open_feedback_label}"' if open_feedback_label else ''}
        )
        """
        feedback = collector.st_feedback(
            feedback_type=feedback_type, metadata=metadata, path=file_name, open_feedback_label=open_feedback_label
        )

    if feedback:
        st.write("Example of the FeedbackCollector's output:")
        st.write(feedback.dict())
        st.download_button("Download this example .json file", feedback.json(), mime="text/json", file_name=file_name)
        st.markdown(
            """
            As you collect feedback, .json files are saved to the 'feedback' folder on your local filesystem.
            """
        )
    st.markdown("***")


def color_cells(val):
    return "background-color: #FF7F7F"  # Light red color

def main():
    st.set_page_config(page_title="Make Predictions", page_icon="📈")
    model = load('/Users/pavelbozmarov/Desktop/trubrics/intern-technical-test-main/model.pickle')

    SPECIES_DICT = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}
    SPECIES_IMAGE_DICT = {
        0: "images/iris_setosa.png",
        1: "images/iris_versicolor.png",
        2: "images/iris_virginica.png",
    }
    st.title("Predict using custom inputs ")

    st.sidebar.title(" 🧪 Enter custom inputs")
    
    # Define sliders for each of the features
    sepal_length = st.sidebar.slider('Sepal length:', 4.3, 7.9, 5.8)
    sepal_width = st.sidebar.slider('Sepal width:', 2.0, 4.4, 3.1)
    petal_length = st.sidebar.slider('Petal length:', 1.0, 6.9, 4.3)
    petal_width = st.sidebar.slider('Petal width:', 0.1, 2.5, 1.3)
    
    # Create a dictionary with the feature names and the selected values
    features = {'sepal_length': sepal_length, 'sepal_width': sepal_width, 
                'petal_length': petal_length, 'petal_width': petal_width}
    
    # Convert the dictionary to a pandas DataFrame and display it
    features_df = pd.DataFrame([features])
    
    # Define data context
    x = pd.read_csv('data/X.csv')
    y = pd.read_csv('data/y.csv')
    pred = model.predict(features_df)[0]
    
    SPECIES_DICT = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}
    # Modification: Convert numerical prediction to species name
    pred_species = SPECIES_DICT[pred]

    st.markdown("***")
    st.write(f'#### 🎰 CUSTOM PREDICTIONS:')
    st.write('Features:', features_df)
    st.write(f'##### Prediction: {pred} ({pred_species})')  # Display both numeric and string labels
    
    # Images of the Iris flowers (You need to replace these with your actual paths or URLs)
    SPECIES_IMAGE_DICT = {
        0: "images/iris_setosa.png",
        1: "images/iris_versicolor.png",
        2: "images/iris_virginica.png",
    }

    # Modification: Display image of predicted species
    image_path = SPECIES_IMAGE_DICT[pred]
    st.image(image_path, caption=pred_species, width=300)
    
    metadata = {"data": features_df.to_dict(), "prediction": pred}
    
    
    st.write('#### meta', metadata)   
    

    # Collecting the feedback
    st.markdown("***")
    collector = FeedbackCollector()

    st.markdown('##### 1 - "Does this prediction look correct?"')
    thumbs_open_feedback = st.radio(
        "Add open feedback?",
        ("No open feedback", "With open feedback"),
        label_visibility="collapsed",
        key="thumbs_radio",
        horizontal=True,
    )
    if thumbs_open_feedback == "With open feedback":
        feedback_example(
            "thumbs", collector=collector, metadata=metadata, open_feedback_label="Please provide a description"
        )
    elif thumbs_open_feedback == "No open feedback":
        feedback_example("thumbs", collector=collector, metadata=metadata)
    else:
        raise NotImplementedError()

    
    st.markdown('##### 2 - "How satisfied are you with this prediction?"')
    faces_open_feedback = st.radio(
        "Add open feedback?",
        ("No open feedback", "With open feedback"),
        label_visibility="collapsed",
        key="faces_radio",
        horizontal=True,
    )
    if faces_open_feedback == "With open feedback":
        feedback_example(
            "faces", collector=collector, metadata=metadata, open_feedback_label="Please provide a description"
        )
    elif faces_open_feedback == "No open feedback":
        feedback_example("faces", collector=collector, metadata=metadata)
    else:
        raise NotImplementedError()

    st.markdown('##### 3 - "Raise a specific issue"')
    feedback_example("issue", collector=collector, metadata=metadata)

    custom_question = "How much do you love this component?"
    st.markdown(f'##### 4 - "{custom_question}"')
    slider = st.slider("Custom feedback slider", max_value=10, value=9)
    submit = st.button("Save feedback")
    code_snippet = """
    from trubrics.integrations.streamlit import FeedbackCollector
import streamlit as st

collector = FeedbackCollector()

slider = st.slider("Custom feedback slider", max_value=10, value=9)
submit = st.button("Save feedback")

if submit and slider:
    collector.st_feedback(
        "custom",
        user_response={
            "How much do you love this component?": slider,
        }
    )
        """
    with st.expander("See code snippet for feedback_type='custom'"):
        st.code(code_snippet)
    if submit and slider:
        feedback_example(
            "custom",
            collector=collector,
            metadata=metadata,
            user_response={
                custom_question: slider,
            },
        )


if __name__ == '__main__':
    main()
