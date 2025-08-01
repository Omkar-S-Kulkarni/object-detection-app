# 🫁 Pneumonia Detection from Chest X-Rays using DenseNet121

A deep learning project that detects pneumonia in chest X-ray images using a **pretrained DenseNet121 model** with PyTorch. This project is modular, well-structured, and suitable for internship or academic portfolios.

---

## 📌 Key Features

- ✅ Transfer learning with `DenseNet121`
- ✅ Binary classification: Normal vs Pneumonia
- ✅ Early stopping to avoid overfitting
- ✅ GPU-compatible (CUDA or CPU fallback)
- ✅ Modular codebase (`train.py`, `test.py`, `model.py`, `utils.py`)
- ✅ Automatic loss visualization and confusion matrix
- ✅ Scalable to >4000 images

---

## 🏗️ Project Structure

```bash
pneumonia-detection/
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   └── utils.py
├── outputs/
├── main.py
├── requirements.txt
└── README.md


data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── validation/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

🚀 How to Run
Step 1: Organize your dataset in the data/ folder
Step 2: Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Step 3: Train & evaluate the model
bash
Copy
Edit
python main.py
This will:

Load your dataset

Train the model with early stopping

Save the best model as outputs/best_model.pth

Plot loss curves

Evaluate on test set

Show confusion matrix

🔧 Training Configuration
Model: DenseNet121 (pretrained)

Loss: BCEWithLogitsLoss()

Optimizer: Adam (lr = 1e-3)

Epochs: 30 (early stopping with patience = 5)

Batch Size: 32

📊 Outputs
📉 Training/validation loss curve

📊 Confusion matrix

✅ Test accuracy

💾 Saved model: outputs/best_model.pth

📈 Example Results
Accuracy: ~92–95% on 4000+ images

Early stopping used to prevent overfitting

Confusion matrix plotted for diagnostics

✨ Future Work
 Add Grad-CAM explainability

 Build a Gradio-based Web UI

 Add hyperparameter tuning with Optuna

 Upload trained model to HuggingFace Hub

📚 References
DenseNet121 paper

Kaggle Dataset

PyTorch Docs


