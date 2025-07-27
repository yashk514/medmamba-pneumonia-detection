# 🔬 Pneumonia Detection with MedMamba Model

A comprehensive Streamlit web application showcasing the development and evaluation of a MedMamba model for classifying pneumonia from medical images, with comparisons against traditional machine learning approaches.

## 🚀 Live Demo

[Your app will be live at: https://yourusername-medmamba-pneumonia-detection-app-xxxxx.streamlit.app]

## 📋 Features

- **MedMamba Architecture**: State-of-the-art neural network for medical image analysis
- **Comprehensive Model Comparison**: 9 different models including Neural Networks, Traditional ML, and Fuzzy Logic
- **Interactive Visualizations**: Training metrics, confusion matrices, ROC curves, and performance comparisons
- **Detailed Analysis**: Model evaluation, Grad-CAM visualizations, and performance insights

## 🏗️ Architecture

### Models Included:
- **Neural Networks**: MedMamba, Graph Fusion, LSTM
- **Traditional ML**: RankSVM (Linear/RBF), GradientBoosting, RandomForest, AdaBoost, KNN, GaussianNB
- **Fuzzy Logic**: Fuzzy Model

### Key Components:
- MedMamba with State Space Model (SSM) architecture
- Data augmentation and preprocessing
- Comprehensive evaluation metrics
- Performance comparison across model types

## 📊 Performance Results

| Model | Accuracy | AUC | Type |
|-------|----------|-----|------|
| Graph Fusion | 98.09% | 99.90% | Neural Network |
| GradientBoosting | 96.37% | 99.58% | Traditional ML |
| RandomForest | 96.37% | 99.44% | Traditional ML |
| LSTM | 95.42% | 99.40% | Neural Network |
| AdaBoost | 95.80% | 99.24% | Traditional ML |
| KNN | 93.89% | 96.26% | Traditional ML |
| MedMamba | 87.60% | 94.91% | Neural Network |
| GaussianNB | 91.22% | 85.38% | Traditional ML |
| Fuzzy Model | 74.24% | 99.46% | Fuzzy Logic |

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/medmamba-pneumonia-detection.git
   cd medmamba-pneumonia-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
medmamba-pneumonia-detection/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── training_metrics.png            # Training performance plots
├── validation_confusion_matrix.png # MedMamba confusion matrix
├── validation_roc_curve.png        # MedMamba ROC curve
├── gradcam_examples.png           # Grad-CAM visualizations
├── model_comparison.png           # RankSVM vs MedMamba comparison
├── extended_model_comparison.png  # All 9 models comparison
├── medmamba_confusion_matrix.png  # MedMamba confusion matrix
├── ranksvm_linear_confusion_matrix.png  # RankSVM Linear confusion matrix
├── ranksvm_rbf_confusion_matrix.png     # RankSVM RBF confusion matrix
├── medmamba_roc_curve.png         # MedMamba ROC curve
├── ranksvm_linear_roc_curve.png   # RankSVM Linear ROC curve
└── ranksvm_rbf_roc_curve.png      # RankSVM RBF ROC curve
```

## 🎯 Key Findings

- **Graph Fusion** achieved the highest performance with 98.09% accuracy
- **Traditional ML models** like GradientBoosting and RandomForest performed exceptionally well
- **MedMamba** shows competitive performance among neural network approaches
- **RankSVM models** achieved higher accuracy but rely on pre-extracted features

## 🔬 Technical Details

### MedMamba Architecture:
- Custom MambaBlock with State Space Model (SSM)
- Enhanced patch embedding with Conv2D layers
- Positional embeddings and skip connections
- Global average pooling with dense classifier head

### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC curves
- Confusion matrices
- Grad-CAM visualizations
- Training/inference time analysis

## 📈 Usage

1. Navigate through the sidebar to explore different sections
2. View model architecture and training details
3. Compare performance across different models
4. Analyze confusion matrices and ROC curves
5. Explore Grad-CAM visualizations for model interpretability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MedMamba architecture for medical image analysis
- PneumoniaMNIST dataset for evaluation
- Streamlit for the web application framework
- All contributors and researchers in the field

---

**Created with ❤️ using Streamlit** 