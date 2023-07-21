import cv2

# Load the pre-trained car detection model (Haar Cascade classifier)
car_classifier = cv2.CascadeClassifier(r'C:\Users\Asus\Desktop\getting started PYTHON\.vscode\VISION_source_codes\CAR PARK OCCUPANCY\cars.xml')

# Load the image
image_filename = r'C:\Users\Asus\Desktop\getting started PYTHON\.vscode\VISION_source_codes\CAR PARK OCCUPANCY\parkphoto.jpeg'
frame = cv2.imread(image_filename)

# Convert the frame to grayscale for Haar Cascade classifier
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect cars in the frame
cars = car_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Create an empty list to store car detections along with confidence values
cars_with_confidence = []

# Loop through the detected cars and store their coordinates and confidence in the list
for (x, y, w, h) in cars:
    # You can set confidence to 1 for Haar Cascades, as they don't provide a confidence score.
    # If you're using a different detection model that provides confidence scores, replace 1 with the actual confidence value.
    confidence = 1
    cars_with_confidence.append(((x, y, w, h), confidence))

# Set the confidence threshold
confidence_threshold =0.5

# Draw rectangles around the detected cars if the confidence is above the threshold
for (x, y, w, h), confidence in cars_with_confidence:
    if confidence > confidence_threshold:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200,100, 50), 2)

# Calculate and display the occupancy status
total_spaces = 3  # Replace this with the total number of parking spaces in your parking lot
empty_spaces = total_spaces - len(cars_with_confidence)
occupancy_status = f"Occupied: {len(cars_with_confidence)}, Empty: {empty_spaces}"
cv2.putText(frame, occupancy_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the frame with car detection and occupancy status
cv2.imshow('Car Parking Occupancy Detection', frame)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
