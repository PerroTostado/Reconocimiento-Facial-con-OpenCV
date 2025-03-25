import cv2
import os

DATA_PATH = 'C:/Users/Kaila/Documents/Reconocimiento Facial/data' # Ruta del dataset de rostros
MODEL_PATH = 'modeloLBPHFace.xml'
HAAR_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

def initialize_face_recognizer():
#"""Inicializa y carga el modelo de reconocimiento facial"""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    return recognizer

def initialize_camera():
#"""Inicializa la captura de video"""
    # Para cámara web:
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # Para archivo de video:
    # cap = cv2.VideoCapture('Video.mp4')
    return cap

def process_faces(frame, face_classifier, recognizer, labels):
#"""Detecta y reconoce rostros en el frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aux_frame = gray.copy()
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = aux_frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (150, 150), interpolation=cv2.INTER_CUBIC)
        
        # Realizar predicción
        label, confidence = recognizer.predict(face_roi)
        
        # Mostrar resultados
        display_recognition_result(frame, x, y, w, h, label, confidence, labels)

    return frame

def display_recognition_result(frame, x, y, w, h, label, confidence, labels):
#"""Muestra el resultado del reconocimiento en el frame"""
    # Umbral de confianza para LBPH (ajustable según necesidad)
    CONFIDENCE_THRESHOLD = 70
    if confidence < CONFIDENCE_THRESHOLD:
        text = f"{labels[label]} ({confidence:.2f})"
        color = (0, 255, 0)  # Verde
    else:
        text = "Desconocido"
        color = (0, 0, 255)  # Rojo

    # Dibujar rectángulo y texto
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, text, (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
def main():
# Cargar nombres de las personas
    person_labels = os.listdir(DATA_PATH)
    print("Personas registradas:", person_labels)

    # Inicializar componentes
    face_recognizer = initialize_face_recognizer()
    face_classifier = cv2.CascadeClassifier(HAAR_CASCADE)
    video_capture = initialize_camera()

    # Bucle principal
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Procesar y mostrar frame
            processed_frame = process_faces(frame, face_classifier, face_recognizer, person_labels)
            cv2.imshow('Reconocimiento Facial', processed_frame)
            
            # Salir con ESC
            if cv2.waitKey(1) == 27:
                break
    finally:
        # Liberar recursos
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()