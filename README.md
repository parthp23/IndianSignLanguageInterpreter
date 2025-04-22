# Indian Sign Language Interpreter (Desktop Version)

A real-time hand gesture recognition system for Indian Sign Language (ISL), designed to help bridge communication gaps for mute individuals. Built with Python, OpenCV, and a custom-trained deep learning model, it converts hand gestures into readable text using a modern desktop interface powered by `customtkinter`.

---

## 📌 Overview

This project captures hand gestures through a webcam, uses **MediaPipe** to extract hand landmarks, and feeds them into a trained TensorFlow model to recognize ISL alphabets. The interpreted letters are assembled into words/sentences and displayed in a clean UI.

---

## ✨ Features

- 🖐️ Real-time gesture recognition for ISL alphabets.
- 🔤 Predicts characters and appends them with hold/cooldown logic.
- 🧠 Uses a CNN model trained on 42 keypoints (21 landmarks × x/y).
- 🖥️ Desktop GUI using `customtkinter` with webcam feed, live predictions, and word suggestions.

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **OpenCV** – Webcam and image processing
- **MediaPipe** – Hand tracking and landmark extraction
- **TensorFlow/Keras** – Gesture recognition model
- **customtkinter** – Modern GUI interface
- **NumPy** – Landmark normalization and handling
- **PyInstaller** – To package the app for distribution

---



![image](https://github.com/user-attachments/assets/4111b7df-4ce9-47bf-a8fa-050083a63acf)
![image](https://github.com/user-attachments/assets/5df06b5a-3631-4ad2-9144-979d8bfce687)

