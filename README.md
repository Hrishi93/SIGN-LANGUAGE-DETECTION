🤟 Real-Time Sign Language Detection System
📌 Project Overview
This project is a Real-Time Sign Language Detection System developed using Computer Vision and Machine Learning. The system detects hand gestures through a webcam and converts them into meaningful text in real time, helping bridge the communication gap between hearing-impaired individuals and others.

The application uses MediaPipe for hand landmark detection and a trained Machine Learning model for gesture classification. The output is displayed through a web-based interface.

🎯 Problem Statement
Communication between hearing-impaired individuals and the general public is often challenging due to the lack of widespread understanding of sign language. Human interpreters may not always be available, leading to communication barriers. This project aims to provide an automated real-time gesture-to-text translation system.

🎯 Objectives
Capture real-time video using a webcam

Detect and extract hand landmarks using MediaPipe

Train a machine learning model for gesture classification

Convert detected gestures into text

Display the output through a user-friendly web interface



🏗 System Architecture
Webcam → MediaPipe Hand Detection → Feature Extraction → ML Model → Text Output

🛠 Technologies Used
Programming Language
Python 3.x



Libraries & Frameworks
MediaPipe

OpenCV

Scikit-learn

NumPy

Pandas

FastAPI



Frontend
HTML

CSS

JavaScript



🧠 Methodology
Data Collection

Hand gesture samples are collected using a webcam.

MediaPipe extracts 21 hand landmarks (X, Y coordinates).

Data is stored in CSV format.

Model Training

Extracted landmark features are used to train a classification model.

The trained model is saved as a .pkl file.

Real-Time Recognition

Live video is captured.

Landmarks are extracted in real time.

The model predicts the gesture.

The predicted word is displayed instantly.



📊 Features
Real-time hand detection

Landmark-based gesture recognition

Fast and lightweight model

Web-based user interface

Fullscreen camera support

Gesture-to-text translation



📂 Project Structure
Sign-Language-Detection/
│
├── dataset/               # Gesture dataset
├── model/                 # Trained ML model
├── static/                # CSS & JS files
├── src/                   # Backend logic
├── app.py                 # FastAPI backend
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation


🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Server
python -m uvicorn app:app --reload
4️⃣ Open in Browser
http://127.0.0.1:8000


📈 Results
Accurate hand landmark detection

Reliable gesture classification

Real-time prediction display

Smooth user interface performance

⚠ Limitations
Supports only predefined gestures

Works best under proper lighting conditions

Single-hand detection

Does not support full sentence formation

🔮 Future Enhancements
Add more gesture classes

Support continuous sentence formation

Integrate text-to-speech conversion

Improve accuracy using deep learning models

Develop mobile application version

🎓 Applications
Assistive communication tools

Educational platforms

Accessibility technologies

Human-computer interaction systems

👩‍💻 Contributors
MOKA DIVYA 
VASAMSETTI BHUVANESWARI
NITTALA SATYA PRAHARINI
SIRIGINADI DIVYA SRI SAI


📜 License
This project is developed for academic purposes.