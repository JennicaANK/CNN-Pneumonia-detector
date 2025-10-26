# CNN-Pneumonia-detector
A deep learning project that trains a CNN model to classify chest X-ray images for pneumonia. Includes analysis of model performance and ethical considerations.
## Authors
- Aye Nyein Kyaw
- Isiah Ketton

## Description of Question and Research Topic
Pneumonia is a serious lung infection that can be identified from chest X-ray images, but diagnosis often requires trained radiologists and may be subject to human error or limited access to specialists. The goal of this project is to train a Convolutional Neural Network (CNN) model to automatically distinguish between chest X-rays showing pneumonia and those that appear normal. Using the Chest X-Ray Images (Pneumonia) dataset from Kaggle, we will explore how deep learning can aid in faster and more consistent image-based diagnostics. We will evaluate our model using accuracy, recall, and ROC-AUC metrics to understand its diagnostic reliability. Finally, we will discuss limitations, including dataset bias and potential ethical considerations in applying AI to clinical settings.

## Project Outline / Plan
**1. Data Preprocessing**
- Load and prepare the dataset using PyTorch’s ImageFolder and DataLoader.
- Apply image augmentation (rotation, flipping, normalization) to reduce overfitting.
- Split into training, validation, and test sets.

**2. Model Construction**
- Build a CNN from scratch using torch.nn.Sequential or subclassing nn.Module.
- Train with cross-entropy loss and Adam optimizer.
- Evaluate with confusion matrix, precision/recall, and ROC-AUC.

**3. Analysis & Validation**
- Plot accuracy/loss curves.
- Use Grad-CAM to visualize attention regions in chest X-rays.
- Discuss misclassifications and the interpretability of the CNN.

**4. Ethical Reflection**
- Address issues of model bias, data imbalance, and the ethical impact of AI-assisted healthcare diagnostics.

## Data Collection Plan

## Model Plans

## Project Timeline
| Week | Task | Milestone |
| :--- | :--- | :--- |
| **Week 9 (10/14–10/20)** | Create a GitHub repo, define a project idea, and assign roles | Project proposal due |
| **Week 10–11** | Data preprocessing and EDA | Preprocessing notebook complete |
| **Week 12–13** | CNN model training (scratch + transfer learning) | Model notebook complete |
| **Week 14–15** | Evaluation, Grad-CAM visualization, and ethical discussion | Analysis notebook complete |
| **Week 16 (12/2–12/4)** | Final presentation | Present results to the class |
| **12/11** | Submit all notebooks and final project files on GitHub | Final due date |

  


