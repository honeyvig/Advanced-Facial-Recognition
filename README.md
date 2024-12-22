# Advanced-Facial-Recognition
Advanced Facial Recognition using Python typically involves the use of deep learning and computer vision libraries to detect and identify faces in images or video streams. One of the most widely used libraries for facial recognition in Python is face_recognition (built on top of dlib). Additionally, libraries such as OpenCV, TensorFlow, and PyTorch can be employed for various stages of the facial recognition pipeline.

Below is an example of how to implement advanced facial recognition using Python with the face_recognition and OpenCV libraries:
Step 1: Install Dependencies

Before running the code, you need to install the necessary Python libraries. You can do this using pip:

pip install face_recognition opencv-python numpy

Step 2: Basic Facial Recognition Script

This example demonstrates how to recognize faces in an image and draw bounding boxes around the faces.

import face_recognition
import cv2
import numpy as np

# Load an image file and convert it to RGB
image = cv2.imread('image.jpg')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Find all face locations in the image
face_locations = face_recognition.face_locations(rgb_image)

# Draw rectangles around each face
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

# Display the result
cv2.imshow("Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

Explanation:

    face_recognition.face_locations(): This function detects all faces in the image and returns their coordinates (top, right, bottom, left).
    cv2.rectangle(): This function is used to draw rectangles around the faces.
    cv2.imshow(): This function shows the image with the drawn bounding boxes.

Step 3: Advanced Facial Recognition (Face Encoding and Matching)

Now, let's expand the functionality to match a detected face against known faces by encoding faces and comparing them.

import face_recognition
import cv2
import numpy as np

# Load images
image_to_recognize = cv2.imread('image_to_recognize.jpg')
known_image = cv2.imread('known_face.jpg')

# Convert images to RGB
rgb_image_to_recognize = cv2.cvtColor(image_to_recognize, cv2.COLOR_BGR2RGB)
rgb_known_image = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)

# Get face encodings
face_encoding_to_recognize = face_recognition.face_encodings(rgb_image_to_recognize)[0]
known_face_encoding = face_recognition.face_encodings(rgb_known_image)[0]

# Compare faces
results = face_recognition.compare_faces([known_face_encoding], face_encoding_to_recognize)

# Output results
if results[0]:
    print("Match found!")
else:
    print("No match found.")

Explanation:

    face_recognition.face_encodings(): This function generates a 128-dimensional vector (encoding) that uniquely represents the face.
    face_recognition.compare_faces(): This function compares the encoding of a detected face with a known encoding and returns True if they match, otherwise False.

Step 4: Real-Time Face Recognition in a Video Stream

In real-world applications, you might want to recognize faces in real-time from a webcam feed. Here's how to use OpenCV to stream video and perform real-time face recognition.

import face_recognition
import cv2

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Load known face images and encode them
known_image = face_recognition.load_image_file('known_face.jpg')
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Create a list of known face encodings and their names
known_face_encodings = [known_face_encoding]
known_face_names = ["Person 1"]

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame from BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

Explanation:

    cv2.VideoCapture(): Captures video from your webcam.
    face_recognition.face_locations() and face_recognition.face_encodings(): Detect faces and create encodings from each frame.
    cv2.rectangle() and cv2.putText(): Draws rectangles around recognized faces and labels them with the name.
    Exit the loop: Press the 'q' key to stop the video stream.

Step 5: Handling Multiple Faces

In some cases, there might be multiple faces in an image or video. You can use a loop to process each face:

import face_recognition
import cv2

# Load an image with multiple faces
image = cv2.imread('multiple_faces.jpg')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Find all face locations and encodings
face_locations = face_recognition.face_locations(rgb_image)
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # Compare with known faces, this can be customized to include multiple known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw a rectangle around each face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

cv2.imshow("Multiple Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

Conclusion

The facial recognition example above uses Python's face_recognition and OpenCV libraries to detect and recognize faces in images and video streams. The script can be extended to handle tasks like:

    Face verification (matching new faces with stored ones).
    Real-time video processing for live detection.
    Advanced face matching based on encodings.

For more advanced use cases, consider using deep learning models and frameworks like TensorFlow, PyTorch, or Dlib to fine-tune and train your models for more accurate recognition under various conditions.
