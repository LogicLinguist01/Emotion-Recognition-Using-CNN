from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from webcame import webcam_emotion_detection, streaming  # Import your webcam emotion detection logic
from Usingvideo import video_emotion_detection  # Import your video file processing logic

app = Flask(__name__)

# Set up the folder to save uploaded videos
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Main route for the index page
@app.route('/')
def index():
    # Stop the webcam when navigating back to the main page
    global streaming
    streaming = False
    return render_template('index.html')

# Route for webcam emotion detection
@app.route('/webcam')
def webcam():
    global streaming
    streaming = True
    return Response(webcam_emotion_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for uploading a video file for emotion detection
@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)  # No file uploaded, return to the form
        file = request.files['file']
        if file.filename == '':  # Empty filename
            return redirect(request.url)
        if file:
            # Save the uploaded file to the specified path
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Redirect to the video streaming page where we can show the video and perform emotion detection
            return redirect(url_for('video_feed', filename=file.filename))
    return render_template('upload.html')

# Route to stream the uploaded video for real-time emotion detection
@app.route('/video_feed/<filename>')
def video_feed(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Check if file exists
    if not os.path.exists(file_path):
        return "Video not found!", 404

    # Stream the video frames for emotion detection
    return Response(video_emotion_detection(file_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
