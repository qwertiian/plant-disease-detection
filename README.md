# 🌿 Plant Disease Detection with Explainable AI (Grad-CAM)

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%7C%20Keras-orange.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Interface](https://img.shields.io/badge/UI-Streamlit-red.svg)
![Explainable AI](https://img.shields.io/badge/XAI-Grad--CAM-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

An end-to-end, interpretable computer vision system engineered for automated crop pathology diagnosis across **38 plant disease categories**. This project integrates two deep learning architectures—a **Custom Convolutional Neural Network (CNN)** trained from scratch and a pre-trained **MobileNetV2** transfer learning model—with **Gradient-weighted Class Activation Mapping (Grad-CAM)** to provide visual explanations for every diagnostic prediction.

---

## 📑 Table of Contents
1. [Overview & Motivation](#-overview--motivation)
2. [Explainable AI (Grad-CAM) Theory](#-explainable-ai-grad-cam-theory)
3. [Deep Learning Model Architectures](#-deep-learning-model-architectures)
4. [Dataset Setup (PlantVillage from Kaggle)](#-dataset-setup-plantvillage-from-kaggle)
5. [Repository Structure](#-repository-structure)
6. [Installation & Setup](#-installation--setup)
7. [Model Training](#-model-training)
8. [Running the Interactive Web App](#-running-the-interactive-web-app)
9. [Diagnostic Interpretation](#-diagnostic-interpretation)

---

## 🔬 Overview & Motivation

Early and precise detection of agricultural plant diseases is critical to safeguarding global food security and minimizing yield losses. While modern deep learning classifiers achieve high diagnostic accuracy, they operate as opaque **"black boxes"**. In real-world agricultural operations, agronomists and farmers cannot act on raw probability scores alone without understanding *why* a particular disease was identified. 

A false diagnosis could lead to:
- Inappropriate chemical fungicide or pesticide application.
- Crop burning or unnecessary destruction of healthy leaves.
- Economic losses and environmental damage.

### The Solution: Interpretable Diagnostics
This system resolves the black-box dilemma by coupling high-capacity convolutional classifiers with **Explainable AI (Grad-CAM)**. For every analyzed leaf, the system outputs:
1. **Predicted Pathology & Host Crop** (e.g., *Tomato - Early Blight*).
2. **Confidence Score Metric** (softmax probability percentage).
3. **Grad-CAM Activation Heatmap**, revealing which morphological features (such as concentric chlorotic rings, necrosis spots, or pustules) drove the model's decision.
4. **Superimposed Visual Overlay**, blending high-activation regions directly onto the leaf image for immediate visual validation.

---

## 🧠 Explainable AI (Grad-CAM) Theory

**Gradient-weighted Class Activation Mapping (Grad-CAM)** utilizes the gradients of any target concept flowing into the final convolutional layer to generate a coarse localization map highlighting the regions that contributed most to the prediction score.

```
Input Image ──► Convolutional Layers ──► [Target Conv Feature Maps A^k] ──► Dense Head ──► Predicted Score (y^c)
                                                   ▲                                            │
                                                   └──────── Backward Gradients (∂y^c / ∂A^k) ──┘
                                                                       │
                                                            Global Average Pooling
                                                                       │
                                                             Weights α_k^c ⊗ A^k
                                                                       │
                                                                 ReLU Operation
                                                                       │
                                                             Grad-CAM Heatmap
```

### Mathematical Formulation

#### 1. Gradient Computation & Feature Importance ($\alpha_k^c$)
Let $A^k \in \mathbb{R}^{u \times v}$ denote the feature map activation of channel $k$ in the final convolutional layer. For a chosen class $c$ with output score $y^c$ (before softmax), the importance weight $\alpha_k^c$ is calculated by computing the spatial global average pooling of the gradients:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$

Where:
- $Z = u \times v$ is the spatial dimensions (height $\times$ width) of the target convolutional feature map.
- $\frac{\partial y^c}{\partial A_{i,j}^k}$ represents the gradient of the score for class $c$ with respect to the activation at spatial position $(i, j)$ in channel $k$.

#### 2. Weighted Sum & Rectified Linear Activation (ReLU)
The final localization map $L_{\text{Grad-CAM}}^c$ is computed as a weighted linear combination of forward activation maps followed by a $\text{ReLU}$ function:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

- **Why ReLU?** We only care about features that positively contribute to the target class $c$ (pixels that increase $y^c$). Features with negative gradients correlate with other competing classes and are filtered out.

#### 3. Normalization & Colormap Overlay
The activation map is scaled to the $[0, 1]$ range:

$$L_{\text{norm}} = \frac{L_{\text{Grad-CAM}}^c - \min(L_{\text{Grad-CAM}}^c)}{\max(L_{\text{Grad-CAM}}^c) - \min(L_{\text{Grad-CAM}}^c)}$$

It is then upsampled to original input dimensions via bicubic interpolation and blended using a **Jet colormap**:
$$\text{Overlay} = \beta \cdot \text{Colormap}(L_{\text{norm}}) + (1 - \beta) \cdot \text{Image}_{\text{RGB}}$$

---

## 🏗 Deep Learning Model Architectures

The project incorporates two complementary architectures to allow trade-off analysis between custom feature engineering and transfer learning:

### 1. Custom CNN Architecture (Trained from Scratch)
Designed to extract visual patterns from raw leaf pixels using standard convolutional blocks:
- **Input Dimension:** $128 \times 128 \times 3$
- **Preprocessing:** Rescaling layer normalizing pixel values from $[0, 255]$ to $[0, 1]$.
- **Block 1:** $\text{Conv2D}(32 \text{ filters}, 3 \times 3, \text{ReLU}) \rightarrow \text{MaxPooling2D}(2 \times 2)$
- **Block 2:** $\text{Conv2D}(64 \text{ filters}, 3 \times 3, \text{ReLU}) \rightarrow \text{MaxPooling2D}(2 \times 2)$
- **Block 3 (Grad-CAM Target):** $\text{Conv2D}(128 \text{ filters}, 3 \times 3, \text{ReLU}, \text{name}=\text{'last\_conv'}) \rightarrow \text{MaxPooling2D}(2 \times 2)$
- **Classifier Head:** $\text{Flatten} \rightarrow \text{Dense}(128, \text{ReLU}) \rightarrow \text{Dropout}(0.5) \rightarrow \text{Dense}(38, \text{Softmax})$

### 2. MobileNetV2 Architecture (Transfer Learning)
Optimized for high accuracy and ultra-low parameter footprint, ideal for edge deployment on mobile or drone agricultural platforms:
- **Pre-trained Backbone:** MobileNetV2 trained on ImageNet with frozen weights.
- **Depthwise Separable Convolutions:** Factorizes standard convolutions into:
  1. A $3 \times 3$ depthwise convolution applied independently per channel.
  2. A $1 \times 1$ pointwise convolution combining channel outputs.
  - This reduces computational cost by approximately:
    $$\frac{1}{N} + \frac{1}{D_k^2} \approx 8\times \text{ to } 9\times \text{ fewer operations}$$
- **Target Conv Layer for Grad-CAM:** `Conv_1` (the final point-wise convolutional layer producing 1280 feature channels).
- **Classifier Head:** $\text{GlobalAveragePooling2D} \rightarrow \text{Dense}(128, \text{ReLU}) \rightarrow \text{Dropout}(0.3) \rightarrow \text{Dense}(38, \text{Softmax})$

---

## 📦 Dataset Setup (PlantVillage from Kaggle)

The model is trained on the benchmark **PlantVillage dataset**, which contains **54,305 curated color images** distributed across **38 classes** (14 crop species and 26 distinct plant diseases + healthy leaves).

### Step 1: Download from Kaggle
Download the dataset archive directly from Kaggle:
👉 **[PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset/data)**

Alternatively, using the official Kaggle CLI:
```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset
```

### Step 2: Extract & Verify Directory Structure
Extract the downloaded archive so that the folder path matches the project configuration:

```text
DL_lab/
└── PlantLeave Dataset/
    └── plantvillage dataset/
        └── color/
            ├── Apple___Apple_scab/
            ├── Apple___Black_rot/
            ├── Apple___Cedar_apple_rust/
            ├── Apple___healthy/
            ├── Blueberry___healthy/
            ├── Cherry_(including_sour)___Powdery_mildew/
            ├── Cherry_(including_sour)___healthy/
            ├── Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
            ├── Corn_(maize)___Common_rust_/
            ├── Corn_(maize)___Northern_Leaf_Blight/
            ├── Corn_(maize)___healthy/
            ├── Grape___Black_rot/
            ├── Grape___Esca_(Black_Measles)/
            ├── Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
            ├── Grape___healthy/
            ├── Orange___Haunglongbing_(Citrus_greening)/
            ├── Peach___Bacterial_spot/
            ├── Peach___healthy/
            ├── Pepper,_bell___Bacterial_spot/
            ├── Pepper,_bell___healthy/
            ├── Potato___Early_blight/
            ├── Potato___Late_blight/
            ├── Potato___healthy/
            ├── Raspberry___healthy/
            ├── Soybean___healthy/
            ├── Squash___Powdery_mildew/
            ├── Strawberry___Leaf_scorch/
            ├── Strawberry___healthy/
            ├── Tomato___Bacterial_spot/
            ├── Tomato___Early_blight/
            ├── Tomato___Late_blight/
            ├── Tomato___Leaf_Mold/
            ├── Tomato___Septoria_leaf_spot/
            ├── Tomato___Spider_mites Two-spotted_spider_mite/
            ├── Tomato___Target_Spot/
            ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
            ├── Tomato___Tomato_mosaic_virus/
            └── Tomato___healthy/
```

---

## 📁 Repository Structure

```text
├── app.py                  # Streamlit web application with interactive Grad-CAM visualization
├── gradcam.py              # Gradient-weighted Class Activation Mapping calculation module
├── train_custom_cnn.py     # Custom CNN model architecture definition and training pipeline
├── train_mobilenet.py      # MobileNetV2 transfer learning model pipeline
├── class_names.json        # 38 target plant disease class labels
├── custom_cnn_model.keras  # Pre-trained Custom CNN model checkpoint
├── mobilenet_model.keras   # Pre-trained MobileNetV2 model checkpoint
├── .gitignore              # Ignores large datasets, caches, and virtual environments
└── README.md               # Technical project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/qwertiian/plant-disease-detection.git
cd plant-disease-detection
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install tensorflow streamlit pillow matplotlib numpy
```

---

## 🚀 Model Training

Both models can be reproduced using the provided training scripts.

### Option A: Train Custom CNN
```bash
python train_custom_cnn.py
```
- Trains the custom 3-block convolutional network.
- Saves model checkpoint to `custom_cnn_model.keras`.

### Option B: Train MobileNetV2 Transfer Learning
```bash
python train_mobilenet.py
```
- Loads the pre-trained ImageNet backbone and fits the custom classification head.
- Saves model checkpoint to `mobilenet_model.keras`.

---

## 🖥 Running the Interactive Web App

Launch the local Streamlit dashboard:

```bash
streamlit run app.py
```

### Interface Walkthrough
1. **Model Architecture Switcher (Sidebar):** Select either **MobileNetV2 (Transfer Learning)** or **Custom CNN (From Scratch)**.
2. **Image Input:**
   - **Upload Image:** Upload any field photograph (`.jpg`, `.jpeg`, `.png`).
   - **Sample Leaves:** Automatically select representative disease samples directly from the local dataset folder.
3. **Diagnostic Report:**
   - **Predicted Condition:** Formatted plant species and disease classification.
   - **Prediction Confidence:** Calibrated prediction confidence bar.
4. **Grad-CAM Visual Breakdown (3-Column Layout):**
   - **Column 1:** Input leaf resized to model dimensions ($128 \times 128$).
   - **Column 2:** Normalized Grad-CAM activation map from the target layer.
   - **Column 3:** Color-coded superimposition highlighting infected tissue.

---

## 📊 Diagnostic Interpretation

| Heatmap Color | Activation Intensity | Biological Interpretation |
| :--- | :--- | :--- |
| **Red / Orange** | **Very High ($\ge 0.8$)** | Primary diagnostic regions; lesions, necrotic halos, fungal rust spots, or chlorosis. |
| **Yellow / Green** | **Moderate ($0.4 - 0.7$)** | Margin boundaries between infected and healthy tissue. |
| **Blue / Cool** | **Low ($\le 0.3$)** | Background or unaffected leaf blade tissue ignored by the classifier. |

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
