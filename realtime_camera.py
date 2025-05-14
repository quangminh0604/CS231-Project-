# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array

# # Load the trained model


# model = load_model('best_model_1.h5')

# # Define emotion labels (based on your dataset)
# emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# # Load Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start webcam
# cap = cv2.VideoCapture(1)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # Extract the face ROI
#         face_roi = gray[y:y+h, x:x+w]
#         # Resize to 48x48 (model input size)
#         face_roi = cv2.resize(face_roi, (48, 48))
#         # Convert to array and normalize
#         face_roi = img_to_array(face_roi) / 255.0
#         face_roi = np.expand_dims(face_roi, axis=0)
#         face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension

#         # Predict emotion
#         predictions = model.predict(face_roi)
#         emotion_index = np.argmax(predictions[0])
#         emotion = emotion_labels[emotion_index]
#         confidence = predictions[0][emotion_index] * 100

#         # Draw rectangle and label on the frame
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         label = f"{emotion} ({confidence:.2f}%)"
#         cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow('Facial Emotion Detection', frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and destroy windows
# cap.release()
# cv2.destroyAllWindows()

from tensorflow.keras.models import load_model

try:
    model = load_model('best_model_1.keras')
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
