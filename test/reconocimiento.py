
import cv2
import face_recognition
import json
import os
import subprocess

# Archivo donde se guardarán los datos
json_path = "data/encodings.json"

# Cargar encodings existentes
if os.path.exists(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            existing = json.load(f)
        except Exception:
            existing = []
else:
    existing = []

known_encodings = [p['encoding'] for p in existing if 'encoding' in p]
known_encodings = [list(map(float, enc)) for enc in known_encodings]
known_names = [p['nombre'] for p in existing if 'nombre' in p]

data = []

cap = cv2.VideoCapture(0)
print("Presiona 'i' para identificar rostro desconocido, 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    rostros_frontales = []
    for idx, ((top, right, bottom, left), encoding) in enumerate(zip(face_locations, face_encodings)):
        w = right - left
        h = bottom - top
        aspect_ratio = w / h if h != 0 else 0
        if 0.8 <= aspect_ratio <= 1.2:
            rostros_frontales.append((top, right, bottom, left, encoding))

    rostro_indices = []
    for idx, (top, right, bottom, left, encoding) in enumerate(rostros_frontales):
        matches = []
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
        if True in matches:
            name = known_names[matches.index(True)]
            color = (0, 255, 0)
            label = f"Conocido: {name}"
        else:
            color = (0, 0, 255)
            label = f"Desconocido #{idx+1} (i para id)"
            rostro_indices.append(idx)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, str(idx+1), (left, bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Reconocimiento en vivo', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('i') and rostro_indices:
        print(f"Rostros desconocidos detectados: {[i+1 for i in rostro_indices]}")
        try:
            idx_str = input(f"¿Qué número de rostro quieres identificar? (elige uno de {[i+1 for i in rostro_indices]}): ")
            idx = int(idx_str) - 1
            if idx in rostro_indices:
                encoding = rostros_frontales[idx][4]
                print("Rostro seleccionado. Ingresa los datos:")
                rut = input("RUT: ")
                nombre = input("Nombre: ")
                otros = input("Otros datos (opcional): ")
                persona = {
                    "rut": rut,
                    "nombre": nombre,
                    "otros": otros,
                    "encoding": encoding.tolist()
                }
                data.append(persona)
                known_encodings.append(encoding.tolist())
                known_names.append(nombre)
                print("Rostro añadido y guardado temporalmente.")
            else:
                print("Índice no válido.")
        except Exception as e:
            print(f"Error: {e}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Guardar en JSON
if data:
    # Crear la carpeta si no existe
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    existing.extend(data)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"Datos guardados en {json_path}")
else:
    print("No se capturaron datos nuevos.")