import cv2
import numpy as np
from keras.models import model_from_json
from mtcnn import MTCNN  # MTCNN for face detection

# Emotion labels dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the emotion model
json_file = open('Model/emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("Model/emotioun_model.weights.h5")
print("Loaded model from disk")

# Initialize MTCNN for face detection
face_detector = MTCNN()

# A flag to control whether the webcam is streaming
streaming = True

def webcam_emotion_detection():
    global streaming
    cap = cv2.VideoCapture(0)

    try:
        while streaming:
            # Read each frame from the webcam
            ret, frame = cap.read()
            if not ret:
                break

            # MTCNN works with color images, so we keep the frame in RGB for detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame using MTCNN
            faces = face_detector.detect_faces(rgb_frame)

            # For each face detected, predict the emotion
            for face in faces:
                (x, y, w, h) = face['box']  # Get the bounding box of the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

                # Extract face ROI and convert to grayscale for emotion prediction
                roi_color_frame = frame[y:y + h, x:x + w]
                roi_gray_frame = cv2.cvtColor(roi_color_frame, cv2.COLOR_BGR2GRAY)  # Convert ROI to grayscale

                # Resize the grayscale face region to 48x48 pixels
                resized_frame = cv2.resize(roi_gray_frame, (48, 48))
                cropped_img = np.expand_dims(np.expand_dims(resized_frame, -1), 0)  # Expand dimensions for the model

                # Predict emotion
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                emotion_label = emotion_dict[maxindex]

                # Display the predicted emotion
                cv2.putText(frame, emotion_label, (x + 5, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as a response for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except GeneratorExit:
        cap.release()
    finally:
        cap.release()
