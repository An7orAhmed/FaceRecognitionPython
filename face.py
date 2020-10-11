import os
import cv2
import numpy as np
import face_recognition

face = []
encode = []
known_faces = []
names = []
x = 0

for filename in os.listdir("faces/"):
    if filename.endswith(".jpg"):
        face.append(face_recognition.load_image_file(os.path.join("faces/", filename)))
        encode.append(face_recognition.face_encodings(face[x])[0])
        known_faces.append(encode[x])
        names.append(filename.replace(".jpg",""))
        print(filename)
        x = x + 1

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    locations = face_recognition.face_locations(rgbFrame)
    encodings = face_recognition.face_encodings(rgbFrame, locations)
    
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_faces, encoding)
            
        name = "unknown!"
        face_distances = face_recognition.face_distance(known_faces, encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = names[best_match_index]
        print("Face detected. " + name)
        
    for (top, right, bottom, left) in locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
    cv2.imshow('webcam LIVE', frame)
    
    if cv2.waitKey(1) & 0xFF == 'q':
        break
        
camera.release()
cv2.destroyAllWindows()

