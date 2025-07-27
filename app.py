import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64 # For embedding images

# Function to encode image to base64 for embedding
@st.cache_data
def get_image_as_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# --- Page Configuration ---
st.set_page_config(
    page_title="Pneumonia Detection with MedMamba",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and Introduction ---
st.title("🔬 Pneumonia Detection with MedMamba Model")
st.markdown("""
This application presents the development and evaluation of a MedMamba model for classifying pneumonia from medical images,
and compares its performance against traditional RankSVM models (Linear and RBF kernels).
""")

# --- Sidebar Navigation (Optional but Recommended) ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Environment & Data",
    "MedMamba Architecture",
    "Model Training",
    "Model Evaluation",
    "MedMamba vs. RankSVM Comparison",
    "Extended Model Comparison",
    "Conclusion"
])

if page == "Overview":
    st.header("Overview")
    st.markdown("""
    This project explores the development and evaluation of a MedMamba model for classifying pneumonia from medical images,
    and compares its performance against traditional RankSVM models (Linear and RBF kernels).
    """)
    st.image("https://images.unsplash.com/photo-1579684385150-b49ef725345d?fit=crop&w=800&q=80", caption="Medical Imaging for Diagnosis")

elif page == "Environment & Data":
    st.header("1. Environment Setup and Data Preparation")
    st.markdown("""
    The notebook begins by importing necessary libraries such as `tensorflow`, `numpy`, `matplotlib`, and `sklearn`.
    It then mounts Google Drive to access the `PneumoniaMNIST` dataset.
    The dataset, stored as a `.npz` file, is loaded, and the image arrays (`X_train`, `X_val`) are preprocessed to have a shape of $(224, 224, 3)$ by expanding and repeating the channel dimension.
    TensorFlow `tf.data.Dataset` objects are created for efficient training and validation.
    The class names are defined as 'normal' and 'pneumonia'.

    **Dataset Shapes:**
    * Training data shape: (4708, 224, 224, 3)
    * Validation data shape: (524, 224, 224, 3)
    """)

elif page == "MedMamba Architecture":
    st.header("2. MedMamba Model Architecture")
    st.markdown("""
    The core of the model is the `MambaBlock` custom Keras Layer. This block incorporates components like
    `LayerNormalization`, `Dense` layers for projections,
    `Conv1D` for local feature extraction, and a custom State Space Model (SSM) for handling sequential data.

    The `create_medmamba_model` function defines the overall MedMamba architecture, which includes:
    * **Data augmentation layers:** `RandomFlip` and `RandomRotation`.
    * **An enhanced patch embedding** using `Conv2D` followed by `LayerNormalization` and `Activation`.
    * `Rearrange` layer for sequence processing.
    * Positional embeddings.
    * Four `MambaBlock`s with skip connections and `Dropout` layers.
    * **An enhanced classifier head** with `GlobalAveragePooling1D`, `Dense` layers, and `Dropout`.
    * A final `Dense` layer with `sigmoid` activation for binary classification.
    """)
    st.subheader("Model Summary")
    st.code("""
Model: "functional"
Layer (type)                 Output Shape              Param #   Connected to
====================================================================================================
input_layer (InputLayer)     (None, 224, 224, 3)       0
____________________________________________________________________________________________________
random_flip (RandomFlip)     (None, 224, 224, 3)       0         input_layer[0][0]
____________________________________________________________________________________________________
random_rotation (RandomRotat (None, 224, 224, 3)       0         random_flip[0][0]
____________________________________________________________________________________________________
conv2d (Conv2D)              (None, 14, 14, 64)        49216     random_rotation[0][0]
____________________________________________________________________________________________________
layer_normalization (LayerNo (None, 14, 14, 64)        128       conv2d[0][0]
____________________________________________________________________________________________________
activation (Activation)      (None, 14, 14, 64)        0         layer_normalization[0][0]
____________________________________________________________________________________________________
rearrange (Rearrange)        (None, 196, 64)           0         activation[0][0]
____________________________________________________________________________________________________
embedding (Embedding)        (None, 196, 64)           12544     tf.range[0][0]
____________________________________________________________________________________________________
add (Add)                    (None, 196, 64)           0         rearrange[0][0],embedding[0][0]
____________________________________________________________________________________________________
mamba_block (MambaBlock)     (None, 196, 64)           48480     add[0][0]
____________________________________________________________________________________________________
dropout (Dropout)            (None, 196, 64)           0         mamba_block[0][0]
____________________________________________________________________________________________________
add_1 (Add)                  (None, 196, 64)           0         add[0][0],dropout[0][0]
____________________________________________________________________________________________________
mamba_block_1 (MambaBlock)   (None, 196, 64)           48480     add_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)          (None, 196, 64)           0         mamba_block_1[0][0]
____________________________________________________________________________________________________
add_2 (Add)                  (None, 196, 64)           0         add_1[0][0],dropout_1[0][0]
____________________________________________________________________________________________________
mamba_block_2 (MambaBlock)   (None, 196, 64)           48480     add_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)          (None, 196, 64)           0         mamba_block_2[0][0]
____________________________________________________________________________________________________
add_3 (Add)                  (None, 196, 64)           0         add_2[0][0],dropout_2[0][0]
____________________________________________________________________________________________________
mamba_block_3 (MambaBlock)   (None, 196, 64)           48480     add_3[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)          (None, 196, 64)           0         mamba_block_3[0][0]
____________________________________________________________________________________________________
add_4 (Add)                  (None, 196, 64)           0         add_3[0][0],dropout_3[0][0]
____________________________________________________________________________________________________
global_average_pooling1d (Gl (None, 64)                0         add_4[0][0]
____________________________________________________________________________________________________
dense_20 (Dense)             (None, 128)               8320      global_average_pooling1d[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)          (None, 128)               0         dense_20[0][0]
____________________________________________________________________________________________________
dense_21 (Dense)             (None, 1)                 129       dropout_4[0][0]
====================================================================================================
Total params: 251,717 (983.27 KB)
Trainable params: 251,717 (983.27 KB)
Non-trainable params: 0 (0.00 B)
    """) # From Source 257, 258

elif page == "Model Training":
    st.header("3. Model Training and Callbacks")
    st.markdown("""
    The MedMamba model is compiled using the Adam optimizer with a learning rate of $1e-4$ and `binary_crossentropy`
    as the loss function, appropriate for binary classification.

    Training is performed for 30 epochs with the following callbacks:
    * `TensorBoard`: For logging training metrics.
    * `EarlyStopping`: To stop training if `val_loss` does not improve for 5 epochs, restoring the best weights.
    * `ModelCheckpoint`: To save the best model based on `val_loss`.
    """)
    st.subheader("Training Performance Visualization")
    st.markdown("""
    The plots below show the training and validation accuracy and loss over epochs.
    * **Accuracy:** Both training and validation accuracy generally increase, with training accuracy reaching over 0.90 and validation accuracy reaching over 0.85 by epoch 30.
    * **Loss:** Both training and validation loss decrease, indicating that the model is learning effectively.
    """)

    st.subheader("Training Metrics Summary (Epoch 30)")
    st.markdown("""
    * **Train Accuracy:** 0.9051
    * **Validation Accuracy:** 0.8760
    * **Train Loss:** 0.2333
    * **Validation Loss:** 0.2769
    """)

    # Assuming you have saved 'training_metrics.png' from your Colab notebook
    # You would need to make this file accessible to your Streamlit app (e.g., in the same directory)
    try:
        training_metrics_b64 = get_image_as_base64("training_metrics.png")
        st.image(f"data:image/png;base64,{training_metrics_b64}", caption="Training and Validation Metrics")
    except FileNotFoundError:
        st.warning("Training metrics plot (training_metrics.png) not found. Please ensure it's in the same directory.")

elif page == "Model Evaluation":
    st.header("5. Model Evaluation")
    st.markdown("The trained MedMamba model is thoroughly evaluated on the validation set.")

    st.subheader("5.1. Confusion Matrix")
    st.markdown("""
    The confusion matrix provides a detailed breakdown of correct and incorrect classifications.
    * **Normal (True Label) vs. Predicted Normal:** 85.9%
    * **Normal (True Label) vs. Predicted Pneumonia:** 14.1%
    * **Pneumonia (True Label) vs. Predicted Normal:** 11.8%
    * **Pneumonia (True Label) vs. Predicted Pneumonia:** 88.2%
    """)
    # Assuming you have saved 'validation_confusion_matrix.png'
    try:
        confusion_matrix_b64 = get_image_as_base64("validation_confusion_matrix.png") # You might need to save this specifically
        st.image(f"data:image/png;base64,{confusion_matrix_b64}", caption="Validation Confusion Matrix (%)")
    except FileNotFoundError:
        st.warning("Validation Confusion Matrix plot not found.")
        # Alternatively, you can reconstruct it if you have the data
        st.markdown("**(Fallback: Displaying raw confusion matrix data if image not found)**")
        cm_data = {
            'True Label': ['normal', 'normal', 'pneumonia', 'pneumonia'],
            'Predicted Label': ['normal', 'pneumonia', 'normal', 'pneumonia'],
            'Percentage': [85.9, 14.1, 11.8, 88.2]
        }
        st.dataframe(pd.DataFrame(cm_data))

    st.subheader("5.2. Classification Report and AUC Score")
    st.markdown("""
    The classification report shows precision, recall, and f1-score for each class, along with overall accuracy.
    * **Normal:** Precision: 0.72, Recall: 0.86, F1-score: 0.78, Support: 135
    * **Pneumonia:** Precision: 0.95, Recall: 0.88, F1-score: 0.91, Support: 389
    * **Overall Accuracy:** 0.88
    * **Validation AUC:** 0.9491
    """)

    st.subheader("5.3. ROC Curve")
    st.markdown("""
    The ROC curve visually represents the model's ability to distinguish between classes.
    The AUC (Area Under the Curve) of 0.95 indicates excellent discriminatory power.
    """)
    # Assuming you have saved 'validation_roc_curve.png'
    try:
        roc_curve_b64 = get_image_as_base64("validation_roc_curve.png") # You might need to save this specifically
        st.image(f"data:image/png;base64,{roc_curve_b64}", caption="Validation ROC Curve")
    except FileNotFoundError:
        st.warning("Validation ROC Curve plot not found.")
        # Fallback to plotting if data is available
        st.markdown("**(Fallback: Displaying a placeholder ROC curve plot)**")
        # In a real scenario, you would have fpr, tpr, and auc from your evaluation script
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.plot([0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 1.0], [0, 0.6, 0.75, 0.8, 0.85, 0.9, 0.93, 0.94, 0.95], label='MedMamba (AUC = 0.95)')
        ax.set_title('Validation ROC Curve (Placeholder)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        st.pyplot(fig)


    st.subheader("5.4. Grad-CAM Visualization")
    st.markdown("""
    Grad-CAM is used to visualize the regions of the input image that are most important for the model's prediction.
    This helps in interpreting why the model made a particular decision. The visualizations below show heatmaps
    superimposed on the X-ray images, highlighting areas relevant to the pneumonia diagnosis.
    """)
    # Assuming you have saved the Grad-CAM examples as a single image
    try:
        gradcam_examples_b64 = get_image_as_base64("gradcam_examples.png") # You would need to save these combined
        st.image(f"data:image/png;base64,{gradcam_examples_b64}", caption="Grad-CAM Heatmaps for Pneumonia Cases")
    except FileNotFoundError:
        st.warning("Grad-CAM examples image (gradcam_examples.png) not found. Please combine and save them.")
        st.markdown("**(Example images from the notebook have been shown in the PDF)**")

elif page == "MedMamba vs. RankSVM Comparison":
    st.header("6. MedMamba vs. RankSVM Comparison")

    st.subheader("6.1. Feature Extraction for RankSVM")
    st.markdown("""
    VGG19, pre-trained on ImageNet, is used as a feature extractor for the SVMs.
    The extracted features are then scaled using `StandardScaler`.
    """)

    st.subheader("6.2. RankSVM Training and Evaluation")
    st.markdown("""
    Two RankSVM models are trained: one with a linear kernel and another with an RBF kernel, both with `class_weight='balanced'`.
    """)

    st.subheader("6.3. Extended Model Comparison Table")
    st.markdown("A comprehensive comparative analysis of all models is presented based on Accuracy, AUC, and Model Type.")
    
    # Extended comparison with all 9 models
    extended_comparison_data = {
        'Model': ['Graph Fusion', 'GradientBoosting', 'Fuzzy Model', 'RandomForest', 'LSTM', 'AdaBoost', 'KNN', 'MedMamba', 'GaussianNB'],
        'Accuracy': [0.980916, 0.96374, 0.742366, 0.96374, 0.954198, 0.958015, 0.938931, 0.875954, 0.912214],
        'AUC': [0.999029, 0.995792, 0.994621, 0.994354, 0.994011, 0.992421, 0.962554, 0.949081, 0.853813],
        'Type': ['Neural Network', 'Traditional ML', 'Fuzzy Logic', 'Traditional ML', 'Neural Network', 'Traditional ML', 'Traditional ML', 'Neural Network', 'Traditional ML']
    }
    extended_comparison_df = pd.DataFrame(extended_comparison_data).set_index('Model')
    st.dataframe(extended_comparison_df)
    
    st.subheader("6.3.1. Original RankSVM vs MedMamba Comparison")
    st.markdown("Detailed comparison of MedMamba with RankSVM models including training and inference times.")
    comparison_data = {
        'Model': ['RankSVM (Linear)', 'RankSVM (RBF)', 'MedMamba'],
        'Accuracy': [0.959924, 0.971374, 0.875954],
        'AUC': [0.992402, 0.994554, 0.949081],
        'Train Time (s)': [3.085528, 6.801479, None], # Train time for MedMamba was not explicitly logged here
        'Inference Time (s)': [0.060229, 0.434430, 15.712903]
    }
    comparison_df = pd.DataFrame(comparison_data).set_index('Model')
    st.dataframe(comparison_df)

    st.subheader("6.4. Confusion Matrices Comparison")
    st.markdown("""
    Below are the confusion matrices for all three models, showing their classification performance on the validation set.
    """)
    
    # Create three columns for the confusion matrices
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            medmamba_cm_b64 = get_image_as_base64("medmamba_confusion_matrix.png")
            st.image(f"data:image/png;base64,{medmamba_cm_b64}", caption="MedMamba Confusion Matrix")
        except FileNotFoundError:
            st.warning("MedMamba confusion matrix not found")
    
    with col2:
        try:
            ranksvm_linear_cm_b64 = get_image_as_base64("ranksvm_linear_confusion_matrix.png")
            st.image(f"data:image/png;base64,{ranksvm_linear_cm_b64}", caption="RankSVM (Linear) Confusion Matrix")
        except FileNotFoundError:
            st.warning("RankSVM Linear confusion matrix not found")
    
    with col3:
        try:
            ranksvm_rbf_cm_b64 = get_image_as_base64("ranksvm_rbf_confusion_matrix.png")
            st.image(f"data:image/png;base64,{ranksvm_rbf_cm_b64}", caption="RankSVM (RBF) Confusion Matrix")
        except FileNotFoundError:
            st.warning("RankSVM RBF confusion matrix not found")

    st.subheader("6.5. ROC Curves Comparison")
    st.markdown("""
    Below are the ROC curves for all three models, showing their ability to distinguish between normal and pneumonia cases.
    """)
    
    # Create three columns for the ROC curves
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            medmamba_roc_b64 = get_image_as_base64("medmamba_roc_curve.png")
            st.image(f"data:image/png;base64,{medmamba_roc_b64}", caption="MedMamba ROC Curve")
        except FileNotFoundError:
            st.warning("MedMamba ROC curve not found")
    
    with col2:
        try:
            ranksvm_linear_roc_b64 = get_image_as_base64("ranksvm_linear_roc_curve.png")
            st.image(f"data:image/png;base64,{ranksvm_linear_roc_b64}", caption="RankSVM (Linear) ROC Curve")
        except FileNotFoundError:
            st.warning("RankSVM Linear ROC curve not found")
    
    with col3:
        try:
            ranksvm_rbf_roc_b64 = get_image_as_base64("ranksvm_rbf_roc_curve.png")
            st.image(f"data:image/png;base64,{ranksvm_rbf_roc_b64}", caption="RankSVM (RBF) ROC Curve")
        except FileNotFoundError:
            st.warning("RankSVM RBF ROC curve not found")

    st.subheader("6.6. Extended Model Performance Visualization")
    st.markdown("""
    Comprehensive comparison of all 9 models showing their accuracy and AUC performance across different model types.
    """)
    
    # Try to load the extended model comparison image
    try:
        extended_model_comparison_b64 = get_image_as_base64("extended_model_comparison.png")
        st.image(f"data:image/png;base64,{extended_model_comparison_b64}", caption="Extended Model Comparison: All 9 Models")
    except FileNotFoundError:
        st.warning("Extended model comparison plot (extended_model_comparison.png) not found. Generating fallback visualization.")
        
        # Create fallback visualization with all 9 models
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Color mapping for model types
        colors = {'Neural Network': 'blue', 'Traditional ML': 'orange', 'Fuzzy Logic': 'green'}
        model_colors = [colors[extended_comparison_df.loc[model, 'Type']] for model in extended_comparison_df.index]
        
        # Accuracy comparison
        bars1 = ax1.bar(extended_comparison_df.index, extended_comparison_df['Accuracy'], color=model_colors)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0.7, 1.0)
        ax1.tick_params(axis='x', rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # AUC comparison
        bars2 = ax2.bar(extended_comparison_df.index, extended_comparison_df['AUC'], color=model_colors)
        ax2.set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('AUC')
        ax2.set_ylim(0.7, 1.0)
        ax2.tick_params(axis='x', rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=model_type) 
                          for model_type, color in colors.items()]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("6.7. Original RankSVM vs MedMamba Visual Comparison")
    st.markdown("""
    Bar plots comparing MedMamba with RankSVM models including training and inference times.
    * **Accuracy and AUC:** RankSVM models generally outperform MedMamba in terms of accuracy and AUC on this dataset.
    * **Inference Time:** RankSVM models have significantly faster inference times compared to MedMamba, especially the Linear SVM.
    """)
    # Assuming you have saved 'model_comparison.png'
    try:
        model_comparison_b64 = get_image_as_base64("model_comparison.png")
        st.image(f"data:image/png;base64,{model_comparison_b64}", caption="Model Comparison: Accuracy, AUC, and Inference Time")
    except FileNotFoundError:
        st.warning("Model comparison plot (model_comparison.png) not found. Please ensure it's in the same directory.")
        # Fallback plot for comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        metrics = ['Accuracy', 'AUC', 'Inference Time (s)']
        for i, metric in enumerate(metrics):
            sns.barplot(x=list(comparison_df.index), y=comparison_df[metric], ax=axes[i])
            axes[i].set_title(metric)
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            if metric == 'Inference Time (s)':
                axes[i].set_yscale('log')
            elif metric in ['Accuracy', 'AUC']:
                axes[i].set_ylim(0.8, 1.0)
        plt.tight_layout()
        st.pyplot(fig)


elif page == "Extended Model Comparison":
    st.header("7. Extended Model Comparison")
    st.markdown("""
    This section provides a comprehensive comparison of all 9 models tested on the pneumonia detection task,
    including Neural Networks, Traditional Machine Learning algorithms, and Fuzzy Logic approaches.
    """)
    
    st.subheader("7.1. Complete Model Performance Table")
    st.markdown("""
    A comprehensive comparison of all models showing their accuracy, AUC, and model type.
    """)
    
    # Extended comparison with all 9 models
    extended_comparison_data = {
        'Model': ['Graph Fusion', 'GradientBoosting', 'Fuzzy Model', 'RandomForest', 'LSTM', 'AdaBoost', 'KNN', 'MedMamba', 'GaussianNB'],
        'Accuracy': [0.980916, 0.96374, 0.742366, 0.96374, 0.954198, 0.958015, 0.938931, 0.875954, 0.912214],
        'AUC': [0.999029, 0.995792, 0.994621, 0.994354, 0.994011, 0.992421, 0.962554, 0.949081, 0.853813],
        'Type': ['Neural Network', 'Traditional ML', 'Fuzzy Logic', 'Traditional ML', 'Neural Network', 'Traditional ML', 'Traditional ML', 'Neural Network', 'Traditional ML']
    }
    extended_comparison_df = pd.DataFrame(extended_comparison_data).set_index('Model')
    st.dataframe(extended_comparison_df)
    
    st.subheader("7.2. Performance Analysis by Model Type")
    st.markdown("""
    **Key Findings:**
    * **Graph Fusion (Neural Network)** achieved the highest performance with 98.09% accuracy and 99.90% AUC
    * **Traditional ML models** like GradientBoosting and RandomForest performed exceptionally well
    * **MedMamba** shows competitive performance among neural network approaches
    * **Fuzzy Model** had the lowest accuracy but still achieved high AUC
    """)
    
    # Performance by type analysis
    type_analysis = extended_comparison_df.groupby('Type').agg({
        'Accuracy': ['mean', 'max', 'min'],
        'AUC': ['mean', 'max', 'min']
    }).round(4)
    st.subheader("7.3. Performance Summary by Model Type")
    st.dataframe(type_analysis)
    
    st.subheader("7.4. Visual Performance Comparison")
    st.markdown("""
    Comprehensive comparison of all 9 models showing their accuracy and AUC performance across different model types.
    """)
    
    # Try to load the extended model comparison image
    try:
        extended_model_comparison_b64 = get_image_as_base64("extended_model_comparison.png")
        st.image(f"data:image/png;base64,{extended_model_comparison_b64}", caption="Extended Model Comparison: All 9 Models")
    except FileNotFoundError:
        st.warning("Extended model comparison plot (extended_model_comparison.png) not found. Generating fallback visualization.")
        
        # Create fallback visualization with all 9 models
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Color mapping for model types
        colors = {'Neural Network': 'blue', 'Traditional ML': 'orange', 'Fuzzy Logic': 'green'}
        model_colors = [colors[extended_comparison_df.loc[model, 'Type']] for model in extended_comparison_df.index]
        
        # Accuracy comparison
        bars1 = ax1.bar(extended_comparison_df.index, extended_comparison_df['Accuracy'], color=model_colors)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0.7, 1.0)
        ax1.tick_params(axis='x', rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # AUC comparison
        bars2 = ax2.bar(extended_comparison_df.index, extended_comparison_df['AUC'], color=model_colors)
        ax2.set_title('Model AUC Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('AUC')
        ax2.set_ylim(0.7, 1.0)
        ax2.tick_params(axis='x', rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=model_type) 
                          for model_type, color in colors.items()]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        st.pyplot(fig)


elif page == "Conclusion":
    st.header("Conclusion")
    st.markdown("""
    The MedMamba model demonstrates strong performance in detecting pneumonia, with a validation accuracy of 88.00% and AUC of 0.9491.
    While the RankSVM models achieved higher accuracy and AUC, they rely on pre-extracted features from a large pre-trained model (VGG19).
    The MedMamba model, being an end-to-end learning architecture, offers a promising alternative, especially in scenarios where feature engineering
    or reliance on large pre-trained backbones might be less desirable. Further optimization of the MedMamba architecture or training parameters
    could potentially bridge the performance gap with RankSVMs.
    """)

    st.markdown("---")
    st.markdown("Created with ❤️ using Streamlit")