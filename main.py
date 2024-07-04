import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
from PIL import Image
import soundfile as sf
import sounddevice as sd
from datasets import load_dataset

# Initialize the DETR model and processor for object detection
processor_detr = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model_detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Initialize the text-to-speech components
processor_tts = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model_tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load the xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Function to convert detected object to speech
def object_to_speech(object_name):
    text = f"A {object_name} object is detected."
    inputs = processor_tts(text=text, return_tensors="pt")
    speech = model_tts.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    audio_file = "detected_object.wav"
    sf.write(audio_file, speech.numpy(), samplerate=16000)
    print(f"Audio generated: {text}")
    # Play the audio file
    data, samplerate = sf.read(audio_file)
    sd.play(data, samplerate)
    sd.wait()  # Wait until the audio is finished playing

# Open a connection to the camera (0 is the default camera)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image for DETR
    inputs_detr = processor_detr(images=image, return_tensors="pt")
    outputs_detr = model_detr(**inputs_detr)

    # Convert outputs (bounding boxes and class logits) to COCO API format
    target_sizes = torch.tensor([image.size[::-1]])
    results_detr = processor_detr.post_process_object_detection(outputs_detr, target_sizes=target_sizes, threshold=0.9)[0]

    # Draw bounding boxes and labels on the frame
    for score, label, box in zip(results_detr["scores"], results_detr["labels"], results_detr["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # Convert box to integers for OpenCV
        box = list(map(int, box))
        label_text = f"{model_detr.config.id2label[label.item()]}: {round(score.item(), 3)}"

        # Draw the bounding box
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # Put the label text above the bounding box
        cv2.putText(frame, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert label_text to speech
        object_to_speech(model_detr.config.id2label[label.item()])

    # Display the frame with bounding boxes and labels
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
