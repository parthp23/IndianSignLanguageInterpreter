import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import threading
import pyttsx3
import numpy as np
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import pandas as pd
import string
import time
import os
import tkinter.font as tkFont # Import the tkinter font module

# Load model and words
model = keras.models.load_model("model.h5")
alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)
with open("customwords.txt", "r") as f:
    custom_words = [line.strip().upper() for line in f if line.strip()]

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Globals
predicted_letters = []
last_gesture = None
hold_start_time = None
hold_duration = 0.7
cooldown_duration = 1.5
last_added_time = 0
camera_active = False
cap = None

# Helper functions
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[min(int(landmark.x * image_width), image_width - 1),
            min(int(landmark.y * image_height), image_height - 1)] for landmark in landmarks.landmark]

def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    temp = [[x - base_x, y - base_y] for x, y in landmark_list]
    temp = list(itertools.chain.from_iterable(temp))
    max_value = max(map(abs, temp))
    return [n / max_value for n in temp]

def speak_sentence():
    engine = pyttsx3.init()
    engine.say(sentence_var.get())
    engine.runAndWait()

def clear_text():
    predicted_letters.clear()
    update_text()

def undo_last():
    if predicted_letters:
        predicted_letters.pop()
        update_text()

def update_text():
    current = ''.join(predicted_letters)
    sentence_var.set(current)
    last_letter_var.set(current[-1] if current else "")
    suggestions = []
    if current:
        prefix = current.split()[-1] if ' ' in current else current
        suggestions = [w for w in custom_words if w.startswith(prefix.upper())][:3]
    for i, word in enumerate(suggestions):
        suggestion_buttons[i].config(text=word, state=tk.NORMAL)
    for j in range(len(suggestions), 3):
        suggestion_buttons[j].config(text="", state=tk.DISABLED)

def choose_suggestion(index):
    current = ''.join(predicted_letters)
    if current:
        prefix = current.split()[-1] if ' ' in current else current
        selected = suggestion_buttons[index].cget("text")
        if ' ' in current:
            space_index = current.rfind(' ')
            new_word = current[:space_index+1] + selected
        else:
            new_word = selected
        predicted_letters.clear()
        predicted_letters.extend(list(new_word) + [' '])
        update_text()

def start_camera():
    global camera_active, cap
    if not camera_active:
        cap = cv2.VideoCapture(0)
        camera_active = True
        threading.Thread(target=video_loop, daemon=True).start()

def stop_camera():
    global camera_active, cap
    camera_active = False
    if cap:
        cap.release()

def video_loop():
    global last_gesture, hold_start_time, last_added_time
    with mp_hands.Hands(model_complexity=0, max_num_hands=2, # Detect up to 2 hands
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while camera_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            x1, y1 = w//2 - 150, h//2 - 150
            x2, y2 = x1 + 300, y1 + 300
            roi = frame[y1:y2, x1:x2]

            # Draw bounding box (green square)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            results = hands.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = calc_landmark_list(roi, hand_landmarks)
                    processed = pre_process_landmark(landmark_list)
                    df = pd.DataFrame(processed).transpose()
                    predictions = model.predict(df, verbose=0)
                    label = alphabet[np.argmax(predictions)]

                    current_time = time.time()
                    if label == last_gesture:
                        if hold_start_time is None:
                            hold_start_time = current_time
                        elif current_time - hold_start_time >= hold_duration:
                            if current_time - last_added_time >= cooldown_duration:
                                predicted_letters.append(label)
                                last_added_time = current_time
                                update_text()
                    else:
                        last_gesture = label
                        hold_start_time = None

            # Draw landmarks and hand connections on ROI
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(roi, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing_styles.get_default_hand_landmarks_style(),
                                            mp_drawing_styles.get_default_hand_connections_style())

            # Show live video
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)

# GUI setup
# Splash Screen with Animation
splash_root = tk.Tk()
splash_root.overrideredirect(True)  # No window border
splash_root.geometry("600x400+400+200")
splash_root.configure(bg="#000000")

# Import required here if not already
from tkinter import ttk

# Splash content
title_label = tk.Label(splash_root, text="", font=("Poppins", 24, "bold"), fg="white", bg="#000000")
title_label.pack(pady=100)

# Progress bar style
style = ttk.Style()
style.theme_use('default')
style.configure("TProgressbar", thickness=10, troughcolor='#1a1a1a', background='#00F0FF', bordercolor='#000000', lightcolor='#00F0FF', darkcolor='#00F0FF')

progress = ttk.Progressbar(splash_root, style="TProgressbar", orient="horizontal", length=400, mode="determinate")
progress.pack(pady=20)

# Animation logic for fade-in text
full_text = "Indian Sign Language Interpreter"
displayed_text = []

def animate_text(i=0):
    if i < len(full_text):
        displayed_text.append(full_text[i])
        title_label.config(text=''.join(displayed_text))
        splash_root.after(80, lambda: animate_text(i + 1))

def fill_progress(value=0):
    if value <= 100:
        progress['value'] = value
        splash_root.after(48, lambda: fill_progress(value + 2))
    else:
        splash_root.after(300, launch_main_app)

def launch_main_app():
    splash_root.destroy()

# Trigger animations
animate_text()
fill_progress()

splash_root.mainloop()


root = tk.Tk()
root.title("Sign Language to Text Conversion")
root.geometry("1280x720")

# Load Poppins Regular font after Tkinter window initialization
poppins_font = tkFont.Font(family="Poppins", size=12, weight="bold")

# Create a main container to center everything vertically
main_container = tk.Frame(root)
main_container.pack(expand=True, fill='both') # Vertically center content

# Layout frames inside the main container
content_frame = tk.Frame(main_container)
content_frame.pack(expand=True)

left_frame = tk.Frame(content_frame)
left_frame.grid(row=0, column=0, padx=10, pady=10)
video_label = tk.Label(left_frame, text="Camera feed", font=poppins_font)
video_label.pack()

center_frame = tk.Frame(content_frame)
center_frame.grid(row=0, column=1, padx=10)
try:
    if os.path.exists("allGestures.png"):
        gesture_img = ImageTk.PhotoImage(Image.open("allGestures.png").resize((300, 300)))
        gesture_label = tk.Label(center_frame, image=gesture_img)
        gesture_label.image = gesture_img
        gesture_label.pack()
    else:
        raise FileNotFoundError("allGestures.png not found")
except Exception:
    tk.Label(center_frame, text="Gesture image not available", font=poppins_font, fg="red").pack()

right_frame = tk.Frame(content_frame)
right_frame.grid(row=0, column=2, padx=10, pady=10)

last_letter_var = tk.StringVar()
tk.Label(right_frame, text="Current Character:", font=poppins_font).pack()
tk.Label(right_frame, textvariable=last_letter_var, font=(poppins_font, 24, "bold"), fg="green").pack(pady=5)

sentence_var = tk.StringVar()
tk.Label(right_frame, text="Constructed Sentence:", font=poppins_font).pack()
tk.Entry(right_frame, textvariable=sentence_var, font=poppins_font, width=30).pack(pady=5)

suggestion_buttons = []
for i in range(3):
    btn = tk.Button(right_frame, text="", font=poppins_font, width=25, state=tk.DISABLED, command=lambda i=i: choose_suggestion(i))
    btn.pack(pady=2)
    suggestion_buttons.append(btn)

btn_frame = tk.Frame(right_frame)
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="Undo", font=poppins_font, command=undo_last).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Clear", font=poppins_font, command=clear_text).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Speak", font=poppins_font, command=speak_sentence).grid(row=0, column=2, padx=5)

cam_btn_frame = tk.Frame(main_container)
cam_btn_frame.pack(pady=10)
tk.Button(cam_btn_frame, text="Start Camera", font=poppins_font, command=start_camera).pack(side=tk.LEFT, padx=10)
tk.Button(cam_btn_frame, text="Stop Camera", font=poppins_font, command=stop_camera).pack(side=tk.LEFT, padx=10)

# Apply dark theme colors
root.configure(bg="#000000")
main_container.configure(bg="#000000")
content_frame.configure(bg="#000000")
left_frame.configure(bg="#000000")
center_frame.configure(bg="#000000")
right_frame.configure(bg="#000000")
btn_frame.configure(bg="#000000")
cam_btn_frame.configure(bg="#000000")

# Style helper for buttons
button_style = {
    "font": poppins_font,
    "bg": "#00F0FF", # Orange background
    "fg": "#000000", # Black text
    "activebackground": "#00F0FF",
    "activeforeground": "#ffffff",
    "relief": tk.FLAT,
    "bd": 0,
    "highlightthickness": 0,
    "padx": 10,
    "pady": 5
}

# Update label and entry text color
for widget in right_frame.winfo_children():
    if isinstance(widget, tk.Label):
        widget.configure(bg="#000000", fg="white")
    if isinstance(widget, tk.Entry):
        widget.configure(bg="#1a1a1a", fg="white", insertbackground="white")

# Update left and center frame background
video_label.configure(bg="#000000", fg="white")

# Apply dark theme to suggestion buttons
for btn in suggestion_buttons:
    btn.configure(**button_style)

# Recreate control buttons with theme
for widget in btn_frame.winfo_children():
    widget.destroy()

tk.Button(btn_frame, text="Undo", command=undo_last, **button_style).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Clear", command=clear_text, **button_style).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Speak", command=speak_sentence, **button_style).grid(row=0, column=2, padx=5)

# Recreate camera buttons with theme
for widget in cam_btn_frame.winfo_children():
    widget.destroy()

tk.Button(cam_btn_frame, text="Start Camera", command=start_camera, **button_style).pack(side=tk.LEFT, padx=10)
tk.Button(cam_btn_frame, text="Stop Camera", command=stop_camera, **button_style).pack(side=tk.LEFT, padx=10)


root.mainloop()