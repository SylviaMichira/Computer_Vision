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

# Draw rectangles around the detected cars
for (x, y, w, h) in cars:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 100, 50), 2)

# Calculate and display the occupancy status
total_spaces = 3  # Replace this with the total number of parking spaces in your parking lot
empty_spaces = total_spaces - len(cars)
occupancy_status = f"Occupied: {len(cars)}, Empty: {empty_spaces}"
cv2.putText(frame, occupancy_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show the frame with car detection and occupancy status
cv2.imshow('Car Parking Occupancy Detection', frame)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
