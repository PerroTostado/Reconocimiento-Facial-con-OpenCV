import cv2
import os
import numpy as np

DATASET_PATH = 'C:/Users/Kaila/Documents/Reconocimiento Facial/data' # Ruta del dataset de rostros
people_list = os.listdir(DATASET_PATH)
print('Personas en el dataset:', people_list)

labels = [] # Lista para almacenar las etiquetas (0, 1, 2...)
faces_data = [] # Lista para almacenar las imágenes de rostros
current_label = 0 # Contador para asignar etiquetas

for person_name in people_list:
    person_path = os.path.join(DATASET_PATH, person_name)
    print(f'\nProcesando rostros de: {person_name}')

    # Leer todas las imágenes de la persona actual
    for image_file in os.listdir(person_path):
        image_path = os.path.join(person_path, image_file)
        print(f'  - Leyendo: {image_file}')
        
        # Cargar imagen en escala de grises
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Almacenar imagen y etiqueta
        faces_data.append(gray_image)
        labels.append(current_label)

    # Incrementar etiqueta para la siguiente persona
    current_label += 1

# Métodos para entrenar el reconocedor
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("\nEntrenando el modelo...")
face_recognizer.train(faces_data, np.array(labels))

MODEL_PATH = 'modeloLBPHFace.xml'
face_recognizer.write(MODEL_PATH)
print(f"Modelo entrenado y guardado en: {MODEL_PATH}")

print("\nResumen del entrenamiento:")
for i, person in enumerate(people_list):
    count = labels.count(i)
    print(f"- {person}: {count} rostros")
