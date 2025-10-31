# 🗣️ SERA — Speech Emotion Recognition Assistance

**Live demo:** https://sera-ksr.streamlit.app

**Contributors:** Vaishnavi Sadul, Veera Varuni Radhakrishnan, Rida Khan 

## 📘 Project Overview
SERA (Speech Emotion Recognition Assistance) is a machine learning project that detects and classifies emotions in human speech.  
It uses advanced audio feature extraction and neural network models to identify emotions such as happiness, sadness, anger, fear, and surprise based on the speaker’s tone and frequency characteristics.

This project aims to empower **neurodivergent communities** by improving emotional understanding in communication, while also enhancing **customer service** interactions with empathy-driven automation.

---

## 💡 Problem Statement
Recognizing emotional content in speech remains a challenge, especially without visual cues.  
SERA addresses this by developing a model that accurately interprets human emotions from audio inputs, regardless of context or accent.

---

## 👥 Stakeholders
- **Neurodivergent Communities** (autism, ADHD, dyslexia)  
- **Customer Service Organizations**  
- **Technology Developers**  
- **Advocacy Groups** for inclusion and accessibility  

---

## 🧠 Data Collection
Datasets used:
- **RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song  
- **TESS** — Toronto Emotional Speech Set  
- **SAVEE** — Surrey Audio-Visual Expressed Emotion  

These were merged, balanced, and standardized to create a diverse dataset capturing multiple emotional tones.

---

## 🔍 Exploratory Data Analysis
- Visualized emotion distribution and gender representation  
- Identified dataset bias towards female, native English speakers  
- Balanced dataset by downsampling overrepresented emotions  
- Dropped gender as a feature due to imbalance  

---

## 🎚️ Feature Extraction
Extracted audio features using **Librosa**, including:
1. **MFCCs (Mel-Frequency Cepstral Coefficients)**  
2. **Chroma MFCCs**  
3. **Mel Spectrogram Frequency**  
4. **Tonnetz (tonal centroid features)**  

All features were scaled and encoded.  
Both **mean** and **standard deviation** values were used for better representation.

---

## ⚙️ Models Used
| Model | Description | Accuracy |
|--------|--------------|-----------|
| **Naive Bayes Classifier** | Baseline probabilistic model | ~52% |
| **CNN (Convolutional Neural Network)** | Deep learning model for feature extraction | ~80% |
| **MLP (Multilayer Perceptron)** | Tuned and cross-validated neural network | **~84% (Best)** |

The **MLP classifier** was selected as the final model for deployment.

---

## 💻 Frontend
Built using **Streamlit**.  
Users can:
- Record their voice directly in the app  
- Receive emotion predictions in real time  

---

## 🚀 Future Improvements
- Expand dataset diversity across gender, race, and language  
- Apply **data augmentation** for robustness  
- Improve model fairness and generalization  

---

## 🧩 Tech Stack
- **Python**  
- **Librosa** — audio processing  
- **Scikit-learn**, **TensorFlow/Keras** — modeling  
- **Matplotlib / Seaborn** — visualization  
- **Streamlit** — web app deployment  

---

## ⚙️ How to Run This Project

### 1. Clone the repository  
```bash
git clone https://github.com/<your-username>/SERA.git
cd SERA
```
### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Streamlit application
```bash
https://sera-ksr.streamlit.app
```

### Once the app opens in your browser:
-Click Record to capture your voice.
-The model extracts features and predicts your emotion in real time.
