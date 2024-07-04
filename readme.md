# Third Eye

Third Eye is an assistive technology designed for visually impaired individuals. It uses state-of-the-art object detection and text-to-speech technologies to identify objects in real-time from a video feed and provide audio descriptions of the detected objects.

## Features

- Real-time object detection using DETR (DEtection TRansformers) from Hugging Face
- Text-to-speech conversion for detected objects using SpeechT5 from Hugging Face
- Plays audio descriptions of detected objects
- User-friendly interface with bounding boxes and labels for detected objects

## Installation

### Prerequisites

- Python 3.7+

### Install Dependencies

1. Clone the repository:

```sh
git clone https://github.com/Tharanitharan-M/Third-Eye---Hugging-Face.git
```

2. Install the required Python packages:

```sh
pip install -r requirements.txt
```

### Requirements File

Ensure you have a `requirements.txt` file with the following contents:

```
torch
transformers
opencv-python
Pillow
soundfile
sounddevice
datasets
```

## Usage

1. Run the main script:

```sh
python main.py
```

2. The application will open a connection to your default camera (usually the built-in webcam) and start detecting objects in real-time.

3. Detected objects will be outlined with bounding boxes, and their names will be displayed on the video feed. An audio description of each detected object will be played.

4. To stop the application, press the `q` key.

## How It Works

- **Object Detection**: The application uses the DETR (DEtection TRansformers) model from Hugging Face's transformers library to detect objects in the video feed. The model processes each frame from the video feed and identifies objects, drawing bounding boxes around them and labeling them with the object's name.
- **Text-to-Speech**: Once an object is detected, the object's name is converted to speech using the SpeechT5 model from Hugging Face's transformers library. The audio description is played, providing feedback to the user about the detected object.

## Project Structure

```
thirdeye/
│
├── main.py               # Main script to run the application
├── requirements.txt      # List of Python dependencies
└── README.md             
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [DEtection TRansformers (DETR)](https://github.com/facebookresearch/detr)
- [SpeechT5](https://github.com/microsoft/speecht5)
- [Hifi-GAN](https://github.com/jik876/hifi-gan)

## Contact

For any questions or suggestions, please contact [tharanimtharan@gmail.com](mailto:tharanimtharan@gmail.com).
