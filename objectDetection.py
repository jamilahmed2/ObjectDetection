import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, OptionMenu, StringVar
from PIL import Image, ImageTk
from pynput.mouse import Controller as MouseController, Button as MouseButton
import pyautogui

# YOLO model files
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# COCO class labels defined directly in the code
classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush","gun","pistol"
]

# Initialize mouse controller
mouse = MouseController()

# Main GUI setup
root = tk.Tk()
root.title("YOLO Game/Object Detection")
root.geometry("300x150")

# Function to start detection
def start_detection(selection):
    root.destroy()  # Close the main menu
    detection_window = tk.Tk()
    detection_window.title(f"{selection} Detection")
    detection_window.geometry("800x600")

    # Video feed label
    video_label = Label(detection_window)
    video_label.pack()

    enabled = False  # Variable to control the bot

    def toggle_bot(event):
        nonlocal enabled
        enabled = not enabled

    detection_window.bind('<f>', toggle_bot)  # Bind the 'f' key to toggle the bot

    def process_frame():
        # Capture video based on selection
        if selection == "Camera":
            cap = cv2.VideoCapture(0)  # Use 0 for the webcam
            ret, frame = cap.read()
            if not ret:
                return
        else:  # Game/Screen detection
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Preprocess the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Process the detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Detected an object
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    # Draw bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = str(classes[class_id])
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if enabled:
                        # Mouse click logic (if necessary)
                        mouse.position = (center_x, center_y)
                        mouse.click(MouseButton.left)

        # Convert the frame to ImageTk format and update the GUI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        # Process the next frame after a small delay
        video_label.after(10, process_frame)

    # Start processing frames
    process_frame()

    # Start the GUI main loop for detection
    detection_window.mainloop()

# GUI for selecting detection mode
detection_mode = StringVar(root)
detection_mode.set("Game/Screen")  # Default value

# Dropdown menu for selecting detection mode
detection_menu = OptionMenu(root, detection_mode, "Game/Screen", "Camera")
detection_menu.pack(pady=20)

# Start button to initiate detection
start_button = Button(root, text="Start Detection", command=lambda: start_detection(detection_mode.get()))
start_button.pack(pady=20)

# Start the main GUI loop
root.mainloop()

cv2.destroyAllWindows()