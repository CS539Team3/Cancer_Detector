import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load models
@st.cache_resource
def load_models():
    return {
        "bladder": joblib.load("Model_Saved/bladder.joblib"),
        "brain": joblib.load("Model_Saved/brain.joblib"),
        "breast": joblib.load("Model_Saved/breast.joblib"),
        "colorectal": joblib.load("Model_Saved/colorectal.joblib"),
        "gastric": joblib.load("Model_Saved/gastric.joblib"),
        "leukemia": joblib.load("Model_Saved/leukemia.joblib"),
        "liver": joblib.load("Model_Saved/liver.joblib"),
        "lung": joblib.load("Model_Saved/lung.joblib"),
        "pancreatic": joblib.load("Model_Saved/pancreatic.joblib"),
        "prostate": joblib.load("Model_Saved/prostate.joblib"),
        "renal": joblib.load("Model_Saved/renal.joblib"),
        "throat": joblib.load("Model_Saved/throat.joblib"),
    }

models = load_models()

# Function to align input data to the model's features
def align_features(input_df, model):
    """
    Align input DataFrame columns with the model's expected features.
    """
    required_features = model.feature_names_in_
    aligned_df = input_df.reindex(columns=required_features, fill_value=0)
    return aligned_df

# Function to dynamically extract top genes
def get_top_genes_dynamic(model, input_features, top_n=10):
    try:
        expected_features = model.feature_names_in_
        coefficients = model.named_steps["classifier"].coef_[0]
        gene_contributions = pd.DataFrame({
            "Gene": expected_features,
            "Contribution": coefficients
        })
    except AttributeError:
        try:
            importances = model.named_steps["classifier"].feature_importances_
            gene_contributions = pd.DataFrame({
                "Gene": expected_features,
                "Contribution": importances
            })
        except AttributeError:
            st.warning("This model does not support feature contribution extraction.")
            return pd.DataFrame(columns=["Gene", "Contribution"])
    
    gene_contributions["AbsContribution"] = np.abs(gene_contributions["Contribution"])
    top_genes = gene_contributions.sort_values(by="AbsContribution", ascending=False).head(top_n)
    return top_genes[["Gene", "Contribution"]]

# Sidebar for input and settings
st.sidebar.title("Gene Prediction Settings")
st.sidebar.markdown(
    """
    🧬 **Upload your gene sample data** and adjust the threshold for cancer prediction.
    - File must be a CSV containing valid gene sample data.
    - Adjust threshold to control sensitivity.
    """
)
uploaded_file = st.sidebar.file_uploader("Upload a Gene Sample (CSV)", type="csv")
threshold = st.sidebar.slider("Threshold for Prediction", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Sidebar for model selection
model_to_analyze = st.sidebar.selectbox(
    "Select a Model to Analyze Top Contributing Genes",
    list(models.keys())
)

if st.sidebar.button("Show Top 10 Genes"):
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            if input_df.empty:
                st.warning("The uploaded file is empty. Please upload a valid CSV file.")
                st.stop()
            if model_to_analyze:
                selected_model = models[model_to_analyze]
                aligned_df = align_features(input_df, selected_model)
                top_genes = get_top_genes_dynamic(selected_model, aligned_df.columns)
                st.markdown(f"### Top 10 Contributing Genes for {model_to_analyze.capitalize()}")
                if not top_genes.empty:
                    st.dataframe(top_genes)
                else:
                    st.info(f"No contributions available for {model_to_analyze.capitalize()}.")
        except pd.errors.EmptyDataError:
            st.error("The uploaded file contains no data or is not a valid CSV. Please upload a valid file.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
    else:
        st.warning("Please upload a CSV file to analyze top contributing genes.")

# Button to display all images
if st.sidebar.button("Display model accuracy"):
    try:
        # First row: Confusion Matrix and Cancer Type Correlation
        st.markdown("## Model Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Confusion Matrix")
            st.image("Result/confusion_matrix.png", use_container_width=False, width=300)
        with col2:
            st.markdown("### Cancer Type Correlation")
            st.image("Result/cancer_type_correlation.png", use_container_width=False, width=300)
        
        # Second row: Precision-Recall Curve and Recall Result
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### Precision-Recall Curve")
            st.image("Result/precision_recall_curve.png", use_container_width=False, width=300)
        with col4:
            st.markdown("### Recall Result")
            st.image("Result/recall_result.png", use_container_width=False, width=300)
        
        # Third row: ROC-AUC Result
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("### ROC-AUC Result")
            st.image("Result/roc_auc.png", use_container_width=False, width=300)
        with col6:
            st.markdown('### Probability distribution')
            st.image('Result/prob_distribution.png', use_container_width=False, width=300)


    except FileNotFoundError as e:
        st.error(f"An image file was not found: {e}")

st.sidebar.markdown("### Download Demo Data Sample")
demo_files = [f for f in os.listdir("Demo_data/") if os.path.isfile(os.path.join("Demo_data/", f))]
selected_demo_file = st.sidebar.selectbox("Select a Demo File to Download", demo_files)

if st.sidebar.button("Download Selected Demo Data"):
    try:
        with open(os.path.join("Demo_data/", selected_demo_file), "rb") as file:
            st.sidebar.download_button(
                label=f"📥 Download {selected_demo_file}",
                data=file,
                file_name=selected_demo_file,
                mime="application/octet-stream"
            )
    except FileNotFoundError:
        st.sidebar.error("Selected file not found. Please check the file path.")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {str(e)}")

# Main UI
st.title("Gene Sample Cancer Prediction Tool")
st.markdown(
    """
    This tool leverages machine learning models to predict cancer based on uploaded gene samples. 
    🧪 **Steps to use:**
    1. Upload your gene sample in CSV format.
    2. Adjust the threshold for prediction (optional).
    3. Click **Run Prediction** to analyze.
    """
)

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)
        if input_df.empty:
            st.warning("The uploaded file is empty. Please upload a valid CSV file.")
            st.stop()
        st.markdown("### Uploaded Sample Data")
        st.dataframe(input_df.head(10))  # Show a preview for better user experience
    except pd.errors.EmptyDataError:
        st.error("The uploaded file contains no data or is not a valid CSV. Please upload a valid file.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
        st.stop()
else:
    st.info("🔼 Upload a CSV file from the sidebar to start.")
    st.stop()

if st.button("Run Prediction"):
    results = {}
    st.markdown("### Running Predictions...")
    progress_bar = st.progress(0)

    # Process sample
    for i, (model_name, model) in enumerate(models.items(), start=1):
        try:
            aligned_sample = align_features(input_df, model)
            proba = model.predict_proba(aligned_sample)[0]
            results[model_name] = proba[1]
        except KeyError as e:
            st.warning(f"⚠️ Missing features for the {model_name} model. Skipping...")
        except Exception as e:
            st.error(f"❌ Error processing with {model_name} model: {str(e)}")
            st.stop()

        progress_bar.progress(i / len(models))

    # Compile results
    results_df = pd.DataFrame([results])
    results_df["max_proba"] = results_df.max(axis=1)
    results_df["predict"] = results_df.idxmax(axis=1)
    results_df["final_predict"] = results_df.apply(
        lambda row: "normal" if row["max_proba"] < threshold else row["predict"],
        axis=1
    )
    
    # Display results
    st.markdown("### Prediction Results")

    # Prepare a copy for display
    results_display = results_df.copy()
    numeric_cols = results_display.select_dtypes(include=["float", "int"]).columns

    for col in numeric_cols:
        results_display[col] = results_display[col].map("{:.2f}".format)

    st.dataframe(results_display)

    # Final prediction
    st.markdown("### Final Prediction")
    st.success(f"The sample is classified as: **{results_df['final_predict'].iloc[0]}**")

    # Download results
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name="prediction_results.csv",
        mime="text/csv"
    )
else:
    st.info("👈 Click **Run Prediction** to analyze your uploaded sample.")