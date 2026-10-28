
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #6a0dad;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-card {
    background-color: #f0f8ff;
    padding: 2rem;
    border-radius: 10px;
    border-left: 5px solid #6a0dad;
    margin: 1rem 0;
}
.feature-slider {
    margin: 1.5rem 0;
}
.confidence-bar {
    height: 20px;
    background-color: #e0e0e0;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.confidence-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    text-align: center;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model(format_type='joblib'):
    """Load the model from the specified format"""
    try:
        if format_type == 'joblib':
            model = joblib.load('models/iris_model.joblib')
        elif format_type == 'pickle':
            with open('models/iris_model.pickle', 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError("Unsupported model format")

        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_model_info():
    """Load model metadata"""
    try:
        with open('models/model_info.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading model info: {e}")
        return None


# Default feature ranges
DEFAULT_RANGES = {
    'sepal_length': {
        'min': 4.0, 'max': 8.0, 'default': 5.8
    },
    'sepal_width': {
        'min': 2.0, 'max': 4.5, 'default': 3.0
    },
    'petal_length': {
        'min': 1.0, 'max': 7.0, 'default': 4.0
    },
    'petal_width': {
        'min': 0.1, 'max': 2.5, 'default': 1.2
    }
}


@st.cache_data
def load_feature_ranges():
    """Load feature ranges for sliders"""
    try:
        with open('models/feature_ranges.json', 'r') as f:
            return json.load(f)
    except Exception:
        # Default ranges if file doesn't exist
        return DEFAULT_RANGES.copy()


# Load data
model_info = load_model_info()
feature_ranges = load_feature_ranges()
model = load_model('joblib')

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    # Model format selection
    model_format = st.radio(
        "Model Format",
        ["joblib", "pickle"],
        help="Choose which serialized model format to use for predictions"
    )

    # Reload model button
    if st.button("🔄 Reload Model"):
        load_model.clear()
        model = load_model(model_format)

        if model is not None:
            st.success(f"Model loaded from {model_format} format!")

    st.divider()

    # Model info
    st.subheader("📊 Model Information")

    if model_info:
        st.write(
            f"**Type:** "
            f"{model_info.get('model_type', 'RandomForest')}"
        )
        st.write(
            f"**Accuracy:** "
            f"{model_info.get('accuracy', 0.96):.1%}"
        )
        st.write(
            f"**Features:** "
            f"{len(model_info.get('feature_names', []))}"
        )
        st.write(
            f"**Classes:** "
            f"{len(model_info.get('target_names', []))}"
        )

    st.divider()

    # Quick actions
    st.subheader("🚀 Quick Actions")

    if st.button("📊 Show Dataset Info"):
        st.session_state.show_dataset_info = True

    if st.button("🎯 Make Prediction"):
        st.session_state.make_prediction = True


# Initialize interaction states
if 'show_dataset_info' not in st.session_state:
    st.session_state.show_dataset_info = False

if 'make_prediction' not in st.session_state:
    st.session_state.make_prediction = False

# Main content
st.markdown(
    '<h1 class="main-header">🌸 Iris Flower Classification</h1>',
    unsafe_allow_html=True
)

# Description
st.markdown("""
This app predicts the species of an Iris flower based on its measurements using a machine le arning model.

Adjust the sliders below to input the flower's characteristics and see the prediction!
""")

# Create input form
col1, col2 = st.columns([2, 1])


def create_slider(label, key, help_text):
    """Create a slider with safe default values."""

    # Use defaults when the JSON file is missing a feature or key
    settings = feature_ranges.get(key, {})
    defaults = DEFAULT_RANGES[key]

    minimum = float(settings.get('min', defaults['min']))
    maximum = float(settings.get('max', defaults['max']))
    default = float(settings.get('default', defaults['default']))

    # Prevent invalid slider configurations
    if minimum >= maximum:
        minimum = defaults['min']
        maximum = defaults['max']

    # Ensure the default is within the slider range
    default = max(minimum, min(default, maximum))

    return st.slider(
        label,
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=0.1,
        help=help_text,
        key=key
    )


with col1:
    st.header("📝 Input Features")

    # Create sliders based on feature ranges
    sepal_length = create_slider(
        "**Sepal Length (cm)**",
        "sepal_length",
        "Length of the sepal in centimeters"
    )

    sepal_width = create_slider(
        "**Sepal Width (cm)**",
        "sepal_width",
        "Width of the sepal in centimeters"
    )

    petal_length = create_slider(
        "**Petal Length (cm)**",
        "petal_length",
        "Length of the petal in centimeters"
    )

    petal_width = create_slider(
        "**Petal Width (cm)**",
        "petal_width",
        "Width of the petal in centimeters"
    )


with col2:
    st.header("📊 Current Values")

    # Display current feature values
    features_df = pd.DataFrame({
        'Feature': [
            'Sepal Length',
            'Sepal Width',
            'Petal Length',
            'Petal Width'
        ],
        'Value (cm)': [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    })

    st.dataframe(
        features_df,
        hide_index=True,
        use_container_width=True
    )

    # Display feature array for prediction
    input_features = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])


# Prediction button
if st.button(
    "🎯 Predict Species",
    type="primary",
    use_container_width=True
):
    st.session_state.make_prediction = True

    if model is not None and model_info is not None:
        try:
            # Make prediction
            prediction = model.predict(input_features)
            prediction_proba = model.predict_proba(input_features)[0]

            # Get predicted class name
            predicted_label = prediction[0]
            target_names = model_info.get('target_names', [])

            if isinstance(predicted_label, (int, np.integer)):
                predicted_index = int(predicted_label)

                if 0 <= predicted_index < len(target_names):
                    predicted_class = target_names[predicted_index]
                else:
                    predicted_class = str(predicted_label)
            else:
                predicted_class = str(predicted_label)

            # Display results
            st.markdown(
                '<div class="prediction-card">',
                unsafe_allow_html=True
            )

            st.markdown("### 📋 Prediction Result")
            st.markdown(
                f"**Predicted Species:** **{predicted_class}**"
            )

            # Show confidence scores with progress bars
            st.markdown("### 📈 Confidence Scores")

            # Get actual model class labels where available
            model_classes = getattr(
                model, 'classes_', range(len(prediction_proba))
            )

            for i, prob in enumerate(prediction_proba):
                class_label = model_classes[i]

                if isinstance(class_label, (int, np.integer)):
                    class_index = int(class_label)

                    if 0 <= class_index < len(target_names):
                        species = target_names[class_index]
                    else:
                        species = str(class_label)
                else:
                    species = str(class_label)

                percentage = float(prob) * 100

                col_prog, col_text = st.columns([3, 1])

                with col_prog:
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill"
                             style="width: {percentage}%;">
                            {percentage:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_text:
                    st.write(f"**{species}**")

            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")

    else:
        st.error(
            "❌ Model could not be loaded. "
            "Please check if the model files exist."
        )


# Additional information
with st.expander(
    "📚 About the Iris Dataset",
    expanded=st.session_state.show_dataset_info
):
    st.markdown("""
    The Iris flower dataset is a classic dataset in machine learning and statistics.

    **Dataset Characteristics:**
    - 150 samples (50 per class)
    - 4 features per sample
    - 3 classes (species)

    **Species:**
    - **Iris Setosa**
    - **Iris Versicolor**
    - **Iris Virginica**

    **Features:**
    1. Sepal length (cm)
    2. Sepal width (cm)
    3. Petal length (cm)
    4. Petal width (cm)

    This model uses a **Random Forest classifier** with an accuracy of approximately 96%.
    """)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit and Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
