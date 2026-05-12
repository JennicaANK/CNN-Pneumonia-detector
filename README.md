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
### Aye Nyein Kyaw
- Responsible for dataset download and preprocessing.  
- Prepare a Jupyter Notebook for image loading, resizing, normalization, and augmentation using `torchvision.transforms`.  
- Perform exploratory data analysis (EDA) — show sample images, class distributions, and identify potential class imbalances.

### Isiah Ketton
- Responsible for verifying dataset structure and creating a reproducible loading pipeline with `DataLoader`.  
- Contribute to data cleaning and ensure consistent labeling between training, validation, and testing folders.  
- Document data provenance and licensing details in the README.



## Model Plans
### Aye Nyein Kyaw
- Build and train a **baseline CNN** from scratch with three convolutional blocks and ReLU activations.  
- Optimize using Adam optimizer, with early stopping and learning rate scheduling.  
- Evaluate results using test accuracy, confusion matrix, and ROC-AUC score.

### Isiah Ketton
- Fine-tune a **pretrained ResNet18** model from `torchvision.models` for performance comparison.  
- Analyze the effect of transfer learning on accuracy and convergence.  
- Use **Grad-CAM** to visualize activation maps and interpret the model’s focus areas.



## Project Timeline
| Week | Task | Milestone |
| :--- | :--- | :--- |
| **Week 9 (10/14–10/20)** | Create a GitHub repo, define a project idea, and assign roles | Project proposal due |
| **Week 10–11** | Data preprocessing and EDA | Preprocessing notebook complete |
| **Week 12–13** | CNN model training (scratch + transfer learning) | Model notebook complete |
| **Week 14–15** | Evaluation, Grad-CAM visualization, and ethical discussion | Analysis notebook complete |
| **Week 16 (12/2–12/4)** | Final presentation | Present results to the class |
| **12/11** | Submit all notebooks and final project files on GitHub | Final due date |



  ### 📚 Data Acknowledgment
This project uses the **Chest X-Ray Images (Pneumonia)** dataset by Paul Mooney, 
originally published on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 
and sourced from the **Guangzhou Women and Children’s Medical Center**.

Dataset License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Source: https://data.mendeley.com/datasets/rscbjbr9sj/2
Citation: Kermany, D.S. et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell (2018).



