<h3>GPU ACCELERATED YOLOv5-OpenVINO</h3>
<p>Note: OpenVINO supports Intel GPUs that is its for use on devices that are equipped with Intel Computing units like Lenovo and others with intel iris(x) and others</p>
# Car Object Detection Using YOLO and OpenVINO accelerated by GPUs (mainly for video inference)

## Step 1: Install the Required Libraries

Clone the repository, install dependencies and `cd` to this local directory for commands in Step 2.

```bash
python -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
### Note

- Make sure you've downloaded the video we're using for testing from here: <https://youtu.be/MNn9qKG2UFI?si=Pt6RE8dt17OV67ne>

Use the commands below to run the

```bash
Convert the yolov5s.pt to onnx and then convert the onnx model to OpenVINO IR format
use the command to convert *.pt model to .onnx format
python export.py --weights path/to/yolov5s.pt --data_type FP16/FP32/INT8 --include onnx

# Quick run
Loading OpenVINO model on GPU device

python inference-gpu.py --input_path "path/to/sample_video.mp4" --target_device "GPU" --media_type "video" --threshold 0.4(video inferencing on GPU)
python inference-gpu.py --input_path "path/to/sample_image.jpg" --target_device "GPU" --media_type "image" --threshold 0.4(image inferencing on GPU)

Loading OpenVINO model on CPU device
python inference-gpu.py --input_path "path/to/sample_video.mp4" --target_device "CPU" --media_type "video" --threshold 0.4(video inferencing on CPU)
python inference-gpu.py --input_path "C:\Users\Moses\Downloads\Video\sample_video.mp4" --target_device "GPU" --media_type "image" --threshold 0.4(image inferencing on CPU) 

For Running with Native yolov5 use these commands below

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
