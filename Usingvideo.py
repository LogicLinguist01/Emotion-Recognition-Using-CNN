import cv2
import numpy as np
from keras.models import model_from_json
from mtcnn import MTCNN  # Import MTCNN for face detection

# Emotion labels dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the model
json_file = open('Model/emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("Model/emotioun_model.weights.h5")
print("Loaded model from disk")

# Initialize MTCNN for face detection
face_detector = MTCNN()

def video_emotion_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MTCNN works with RGB images, so convert BGR (OpenCV default) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame using MTCNN
        faces = face_detector.detect_faces(rgb_frame)

        for face in faces:
            (x, y, w, h) = face['box']  # Get the bounding box of the face

            # Draw a rectangle around each detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # Extract the region of interest (face) from the original frame
            roi_color_frame = frame[y:y + h, x:x + w]

            # Convert the face ROI to grayscale for emotion prediction
            roi_gray_frame = cv2.cvtColor(roi_color_frame, cv2.COLOR_BGR2GRAY)

            # Resize the face region to 48x48 pixels, as expected by the emotion model
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # Predict the emotion
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))

            # Put the emotion label on the image
            cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Convert the frame to bytes and yield it to the response
        frame = buffer.tobytes()

        # Yield the frame for the browser to display it
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()
