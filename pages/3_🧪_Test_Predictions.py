import streamlit as st
import pandas as pd
from joblib import load

from trubrics.integrations.streamlit import (
    FeedbackCollector,
    generate_what_if_streamlit,
)

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
    model = load('model.pickle')

    SPECIES_DICT = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virginica'}
    SPECIES_IMAGE_DICT = {
        0: "images/iris_setosa.png",
        1: "images/iris_versicolor.png",
        2: "images/iris_virginica.png",
    }
    st.title("Predict using test dataset ")

    st.markdown("#### See Misclassified Samples")

    # Checkbox options for 'train' and 'test'
    check_train = st.checkbox('Train')

    # Check if 'train' checkbox is checked
    if check_train:
        # Load and display train misclassified samples
        st.write('###### Train Misclassified Samples:')
        df_train_misclassified = pd.read_csv('data/train_misclassified.csv')
        st.dataframe(df_train_misclassified.style.set_precision(1).applymap(color_cells))
    
    check_test = st.checkbox('Test')
    # Check if 'test' checkbox is checked
    if check_test:
        # Load and display test misclassified samples
        st.write('###### Test Misclassified Samples:')
        df_test_misclassified = pd.read_csv('data/test_misclassified.csv')
        df_test_misclassified = df_test_misclassified.applymap(lambda x: round(x, 1) if isinstance(x, (float)) else x) # Round float values
        st.dataframe(df_test_misclassified.style.set_precision(1).applymap(color_cells))

    st.markdown("***")
    
    # Load test data
    df_test = pd.read_csv('data/df_test.csv')

    # Display test data
    st.subheader("📊 Test Data:")
    st.dataframe(df_test)

    # Get user input for row selection
    row_number = st.number_input("Enter a row number to use its features for prediction:", min_value=0, max_value=len(df_test)-1, step=1)
    
    selected_features = df_test.iloc[row_number, :-1]  # All columns except last
    selected_features = selected_features.to_frame().T
    st.write('Selected features:', selected_features)

    true_label = df_test.iloc[row_number, -1]
    prediction = model.predict(selected_features)[0]

    st.write(f'#### 🎰 TEST PREDICTIONS:')


    # Display the prediction and true label above images
    col1, col2 = st.columns(2)

    with col1:
            image_path = SPECIES_IMAGE_DICT[prediction]

            st.write(f'##### Prediction: {prediction} ({SPECIES_DICT[prediction]})')
            
            st.image(image_path, width=300)

    with col2:
            true_label_image_path = SPECIES_IMAGE_DICT[true_label]

            st.write(f'##### True Label: {true_label} ({SPECIES_DICT[true_label]})')
            
            st.image(true_label_image_path,width=300)
    
    
    metadata = {"data": selected_features.to_dict(), "prediction": prediction,"true_label": true_label}
    
    
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
