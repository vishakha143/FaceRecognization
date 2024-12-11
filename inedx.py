import os
import numpy as np
import joblib
import face_recognition
import cv2

#  Load known face encodings
face_encodings = []
names = []

# Load images from the 'face' directory
for image in os.listdir('face'):
    image_path = f"face/{image}"
    resident = face_recognition.load_image_file(image_path)
    
    # Get face encodings
    face_encodings_list = face_recognition.face_encodings(resident)
    if face_encodings_list:  # Check if any face encodings were found
        face_encoding = face_encodings_list[0]
        face_encodings.append(face_encoding)
        names.append(image.split('.')[0])  # Get name without extension

# Save the names and encodings to a file
joblib.dump([names, face_encodings], 'face_encodings.jl')

#  Function to check an image for recognized faces
def check_image(image_path):
    visitors = face_recognition.load_image_file(image_path)
    visitors_face_encodings = face_recognition.face_encodings(visitors)
    
    names, face_encodings = joblib.load('face_encodings.jl')
    output = []
    
    for visitors_face_encoding in visitors_face_encodings:
        results = face_recognition.compare_faces(face_encodings, visitors_face_encoding)
        if any(results):
            person = names[np.argmax(results)]
            output.append(person)
    
    return output, len(visitors_face_encodings) - len(output)

#  Check a test image
lila = 'test/pseudo.jpg'
result = check_image(lila)

if result[1] > 0:
    print('There is an unknown individual at the door.')
else:
    print(f"Welcome, {result[0][0]}")  # Assuming at least one person is recognized

#  Function to visualize detected faces
def visualize_faces(image_path):
    img = cv2.imread(image_path)
    face_locations = face_recognition.face_locations(img)

    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Visualize faces in the test image
visualize_faces(lila)