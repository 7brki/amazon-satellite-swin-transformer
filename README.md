\# 🌍 Amazon Rainforest Satellite Imagery Classification using Swin Transformer \& Asymmetric Loss (ASL)



\[!\[Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/brkiii/amazon-satellite-radar)

\[!\[Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat\&logo=pytorch\&logoColor=white)](https://pytorch.org/)



An end-to-end multi-label remote sensing and spatial data science project designed to detect 17 distinct atmospheric and ecological conditions in the Amazon rainforest simultaneously from satellite imagery.



\---



\## 🚀 Live Demo

The trained model is deployed on Hugging Face Spaces with a clean Gradio web interface. You can upload any multi-spectral or standard RGB satellite crop to see the model infer in real-time.



👉 \*\*\[Live Web Interface Link](https://huggingface.co/spaces/brkiii/amazon-satellite-radar)\*\*



\---



\## 🛠️ Technical Architecture \& Innovations



Unlike traditional CNN approaches (e.g., ResNet, EfficientNet) trained on generic consumer datasets like ImageNet, this architecture leverages state-of-the-art vision mechanisms optimized for complex spatial context:



\### 1. Spatial Attention (Swin Transformer V1)

Utilizes `swin\_tiny\_patch4\_window7\_224` as the backbone feature extractor. The shifted window attention mechanism effectively captures long-range spatial correlations and geometric structures (such as linear road patterns, river bends, and blocky agricultural boundaries) that localized convolutional kernels often fail to extract.



\### 2. Asymmetric Loss (ASL) for Severe Class Imbalance

Standard Binary Cross Entropy (BCE) fails heavily on multi-label earth observation data due to the dominance of frequent classes (e.g., `primary`, `clear`) over rare but critical ecological indicators (e.g., `slash\_burn`, `blow\_down`, `artisinal\_mine`). 

By implementing \*\*Asymmetric Loss (ASL)\*\*, the network dynamically down-weights easy negative spatial samples, forcing the gradient descent to focus entirely on learning high-frequency details of rare classes.



\---



\## 📈 Performance Metrics



The model achieves highly competitive validation scores within a lightweight 2-epoch training constraint, demonstrating immediate stability and generalizability:



\- \*\*Kaggle Leaderboard Prediction (F2-Score):\*\* `0.92299`

\- \*\*Primary/Clear Accuracy:\*\* `>94%`



\### Real-world Generalization Test:

When evaluated on Out-of-Distribution (OOD) textbook remote sensing imagery outside the original Kaggle dataset, the model correctly extracted multi-label features: `clear` (94%), `road` (93%), `habitation` (90%), `primary` (85%), and `agriculture` (82%), confirming robust spatial understanding over mere pixel memorization.



\---



\## 📁 Repository Structure



```text

├── app.py                # Gradio application script for deployment

├── requirements.txt      # Deployment and local dependencies

├── README.md             # Core project documentation

└── notebooks/

&#x20;   └── amazon\_satellite\_analysis.ipynb  # End-to-end training and optimization notebook

```



\---



\## 📦 Installation \& Local Usage



1\. Clone the repository:

&#x20;  ```bash

&#x20;  git clone \[https://github.com/YOUR\_GITHUB\_USERNAME/amazon-satellite-swin-transformer.git](https://github.com/brkiii/amazon-satellite-swin-transformer.git)

&#x20;  cd amazon-satellite-swin-transformer

&#x20;  ```



2\. Install dependencies:

&#x20;  ```bash

&#x20;  pip install -r requirements.txt

&#x20;  ```



3\. Run the Gradio interface locally:

&#x20;  ```bash

&#x20;  python app.py

&#x20;  ```



\---



\## 🔬 Dataset Acknowledgements

The dataset used is from the \*\*"Planet: Understanding the Amazon from Space"\*\* Kaggle competition. It contains high-resolution satellite imagery chips covering the Amazon basin, manually labeled by remote sensing experts into 17 co-occurring classes spanning atmospheric conditions, land cover, and land use types.

