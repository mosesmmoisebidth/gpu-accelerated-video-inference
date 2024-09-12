# Car Object Detection Using YOLO

## Step 1: Install the Required Libraries

Clone the repository, install dependencies and `cd` to this local directory for commands in Step 2.

```bash
python -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 2: Run the Car Object Counting Using YOLO

Make sure you run the backend server first to allow you to send data to the server from the detector in real-time. Here are the basic commands for running the inference:

### Note

- Make sure you've downloaded the video we're using for testing from here: <https://youtu.be/MNn9qKG2UFI?si=Pt6RE8dt17OV67ne>

Use the commands below to run the

```bash
# Quick run
python main.py --source "path/to/video.mp4" --view-img

# If you want to save results
python main.py --source "path/to/video.mp4" --save-img --view-img

# If you want to run model on CPU
python main.py --source "path/to/video.mp4" --save-img --view-img --device cpu

# If you want to change model file
python main.py --source "path/to/video.mp4" --save-img --weights "path/to/model.pt"

# If you want to detect specific class (first class and third class)
python main.py --source "path/to/video.mp4" --classes 0 2 --weights "path/to/model.pt"

# If you don't want to save results
python main.py --source "path/to/video.mp4" --view-img
```

## Usage Options

- `--source`: Specifies the path to the video file you want to run inference on.
- `--device`: Specifies the device `cpu` or `0`
- `--save-img`: Flag to save the detection results as images.
- `--weights`: Specifies a different YOLOv8 model file (e.g., `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`).
- `--classes`: Specifies the class to be detected
- `--line-thickness`: Specifies the bounding box thickness
- `--region-thickness`: Specifies the region boxes thickness
- `--track-thickness`: Specifies the track line thickness
