import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json
import numpy as np 
import cv2   

# Modelos
facetracker = model_from_json(open("facetracker.json", "r").read())
facetracker.load_weights('facetracker.h5')

emotion_model = model_from_json(open("emotion_detection.json", "r").read())
emotion_model.load_weights('emotion_detection.h5')


skip_frames = 2 # Hacer una inferencia cada 2 frames

width, height = 0, 0
selected_model = 'facetracker'  # Modelo seleccionado inicialmente

# Función para manejar los clics del mouse
def click_event(event, x, y, flags, param):
    global selected_model
    if event == cv2.EVENT_LBUTTONDOWN:
        # Verifica si el clic está dentro de los límites del botón izquierdo (modelo de facetracker)
        if x < width // 2 and height - 130 < y < height - 30:
            selected_model = 'facetracker'
        # Verifica si el clic está dentro de los límites del botón derecho (modelo de emotion_detection)
        elif x > width // 2 and height - 130 < y < height - 30:
            selected_model = 'emotion'

# Función para la lógica del modelo de facetracker
def facetracker_logic(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = np.array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = np.expand_dims(img_pixels, axis=-1)
        img_pixels = img_pixels.astype(np.float32) / 255.0  # Convert to float32 and perform division

        cv2.putText(frame, 'Face', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Función para la lógica del modelo de emotion_detection
def emotion_detection_logic(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = np.array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = np.expand_dims(img_pixels, axis=-1)
        img_pixels = img_pixels.astype(np.float32) / 255.0 

        predictions = emotion_model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Función principal
def main():
    global width, height, selected_model

    cap = cv2.VideoCapture(-1)  # Inicializar la cámara
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ancho de la ventana de la cámara
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Alto de la ventana de la cámara

    cv2.namedWindow('Model Selector')  # Creamos una ventana
    frame_count = 0
    while True:
        ret, frame = cap.read()  # Leemos un frame de la cámara
        frame_count += 1

        if frame_count % skip_frames != 0:
            continue  # Saltamos frames para reducir latencia

        if selected_model == 'facetracker':
            frame = facetracker_logic(frame)
        elif selected_model == 'emotion':
            frame = emotion_detection_logic(frame)

        # Botón izquierdo: Modelo de facetracker
        cv2.rectangle(frame, (0, height - 130), (width // 2, height - 30), (255, 0, 0), -1)
        cv2.putText(frame, 'Face Recognition', (10, height - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Botón derecho: Modelo de emotion_detection
        cv2.rectangle(frame, (width - width // 2, height - 130), (width, height - 30), (0, 255, 0), -1)
        cv2.putText(frame, 'Emotions Recognition ', (width // 2 + 10, height - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Model Selector', frame)  # Mostramos la ventana con los botones

        cv2.setMouseCallback('Model Selector', click_event)  # Llamamos a la función click_event cuando se hace clic en la ventana

        cv2.moveWindow('Model Selector', 0, 720 - height)  


        if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
            break

    cap.release()  
    cv2.destroyAllWindows()  

if __name__ == "__main__":
    main()
