import cv2
import dlib
import numpy as np
from keras.models import load_model

# Load the pre-trained model for classifying emotion 
model = load_model(r'C:\Users\Asus\Desktop\getting started PYTHON\.vscode\VISION_source_codes\FACIAL EMOTION RECOGNITION\emotion_model.hdf5')

# Load the shape predictor model
predictor_path = r'C:\Users\Asus\Desktop\getting started PYTHON\.vscode\VISION_source_codes\FACIAL EMOTION RECOGNITION\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# Load the image
image = cv2.imread(r'C:\Users\Asus\Desktop\sylvia.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a dlib detector
detector = dlib.get_frontal_face_detector()

# Detect faces in the image
faces = detector(gray)

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

for face in faces:
    # Get the facial landmarks
    shape = predictor(gray, face)

    # Extract the coordinates of the facial landmarks
    landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)])

    # Extract the face ROI (Region of Interest)
    (x, y, w, h) = cv2.boundingRect(landmarks)
    face_roi = gray[y:y+h, x:x+w]

    # Resize the face ROI to match the input size of the model
    face_roi = cv2.resize(face_roi, (64, 64))

    # Normalize the face ROI
    face_roi = face_roi / 255.0

    # Reshape the face ROI to match the input shape of the model
    face_roi = np.reshape(face_roi, (1, 64, 64, 1))

    # Make emotion prediction using your pre-trained model
    emotion_pred = model.predict(face_roi)
    emotion_label = emotion_labels[np.argmax(emotion_pred)]

    # Draw a rectangle around the face and display the emotion label
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Display the image with emotion labels
cv2.imshow('Facial Emotion Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

