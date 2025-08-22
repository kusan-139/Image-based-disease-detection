# 🩺 Image-Based Disease Detection  

This project detects diseases from **chest X-ray images** using pre-trained **DenseNet121 models**.  

## 📂 Project Structure  

```
IMAGE BASED DISEASE DETECTION/
│
├── chest_xray/                # Dataset (training / testing images)
├── plots/                     # Saved training plots (accuracy, loss, etc.)
├── venv/                      # Virtual environment (dependencies installed here)
│
├── app.py                     # Main Streamlit app script
├── App.bat                    # Batch file to launch app in browser
├── Run.bat                    # Batch file to load models
├── Run_App.bat                # Combined runner (if needed)
├── setup.ps1                  # PowerShell setup script
│
├── densenet121_base.h5        # Pre-trained DenseNet121 (base version)
├── densenet121_finetuned.h5   # Fine-tuned DenseNet121 (better accuracy)
│
├── Model.py                   # Model architecture & training script
├── DiseaseDetector.spec       # PyInstaller spec file for app packaging
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)

---

## ⚙️ Setup Instructions  

1. **Clone the repository / extract files**  
   
   git clone <repo-link>
   cd "IMAGE BASED DISEASE DETECTION"
   

2. **Create virtual environment (optional, recommended)**  
   
   python -m venv venv
   venv\Scripts\activate   # On Windows
   

3. **Install requirements**  
   
   pip install -r requirements.txt
  
If pip is not recognized

Use Python with -m pip, Instead of pip install, run:

python -m pip install -r requirements.txt

---

##  Running the Application  

There are **two ways to run the app** depending on your needs:  

### 1. Load model and start app (automatic)  
Run the provided batch files in sequence:  

- **Step 1:** Run the model loader  
 
  Run.bat

  This will **load the pre-trained `.h5` models** (`densenet121_base.h5` or `densenet121_finetuned.h5`).  

- **Step 2:** Launch the web app  
  
  App.bat
 
  This will open the **Streamlit app** in your default browser.  

---

### 2. Train model from scratch (optional)  
If you want to train a new model:  


python Model.py


This will generate a new `.h5` file that can later be loaded using `Run.bat`.



## 📌 Notes  

- Use `densenet121_finetuned.h5` for **higher accuracy**.  
- Make sure the `chest_xray` dataset is available before training.  
- For just running the app, **you don’t need to retrain** — simply use the pre-trained `.h5` files.  

---

✅ With this, your workflow is:  
**Train (optional) → Run.bat (load model) → App.bat (launch in browser).**
