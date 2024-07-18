import cv2
import os
import face_recognition
import time

# Path to the reference image in the images folder
reference_image_path = 'images/profile-headshot.jpg'

# Load the reference image
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Load the pre-built Haar Cascade model for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def accessWebcam():
    # Access the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Allow the webcam some time to initialize
    time.sleep(2)
    
    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()
        
        # Check if the frame was read successfully
        if not ret:
            print("Error: Failed to capture image from webcam")
            break
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_classifier.detectMultiScale(gray, 1.1, 5)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face region from the frame
            face_region = frame[y:y+h, x:x+w]
            
            # Resize the face region for face recognition (optional but recommended)
            face_region_resized = cv2.resize(face_region, (128, 128))
            
            # Encode the detected face
            face_encoding = face_recognition.face_encodings(face_region_resized)
            
            # Compare the detected face encoding with the reference face encoding
            if len(face_encoding) > 0:
                match = face_recognition.compare_faces([reference_encoding], face_encoding[0], tolerance=0.5)
                
                # If match is found, display the name on the bounding box
                if match[0]:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Raymond", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Display the frame with face detections
        cv2.imshow("Face Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start the webcam feed
accessWebcam()
