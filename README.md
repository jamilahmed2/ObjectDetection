
# YOLO Object Detection with GUI for Game/Screen and Camera

This project implements object detection using YOLO (You Only Look Once) in Python, with a simple GUI interface to choose between game/screen or camera-based detection. The system uses YOLOv3 for object detection and allows users to detect objects such as cars, people, and more in real-time from games, screens, or a connected camera.

## Features

- Real-time object detection using YOLOv3.
- GUI interface to select detection mode: either from a live webcam feed or game/screen capture.
- Supports various COCO dataset objects such as cars, people, buses, and more.
- Click-based interaction when detecting objects on the screen.

## Requirements

To get started with this project, ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

### Required Libraries

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Tkinter (for the GUI)
- PIL (`Pillow`)
- PyAutoGUI (for screen capture)
- Pynput (for mouse control)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/yolo-game-screen-detection.git
cd yolo-game-screen-detection
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Download the YOLOv3 weights and configuration files.

## Download Pre-trained YOLO Weights

You can download the pre-trained YOLOv3 weights using the links below:

- **YOLOv3 Weights**: [Download YOLOv3 weights](https://pjreddie.com/media/files/yolov3.weights)
- **YOLOv3 Configuration**: [Download YOLOv3 config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

Once downloaded, place the weights and configuration files in the project folder or adjust the paths in the code if stored elsewhere.

## How to Use

1. Launch the GUI to select the detection mode (Game/Screen or Camera):

```bash
python main.py
```

2. In the GUI:
   - Choose between **Game/Screen** or **Camera** for the source of object detection.
   - Press **Start Detection** to begin real-time object detection.
   - For **Game/Screen**, the system will capture the current screen for detection.  
   - For **Camera**, it will use your webcam feed.

3. Press the `f` key during detection to toggle click-based interaction.

## Sample Code

This project utilizes OpenCV, YOLOv3, and PyAutoGUI for screen and video capture. Below is an example of how the YOLO object detection process is handled in the code:

```python
# YOLO model files
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define object classes (COCO dataset)
classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", ...]
```

## Customization

- To adjust the objects detected, you can modify the `classes` list in the code to include only the objects you are interested in detecting.
- Adjust the detection confidence threshold in the code to filter out low-confidence detections:
  
```python
if confidence > 0.5:
    # Process detected object
```

## Acknowledgments

This project uses the YOLO implementation by [Joseph Redmon](https://pjreddie.com/darknet/yolo/) and integrates various libraries such as OpenCV, PyAutoGUI, and Tkinter.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
