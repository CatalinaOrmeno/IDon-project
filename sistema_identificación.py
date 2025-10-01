import cv2
import face_recognition
import json
import os
import numpy as np
from math import atan2, degrees

# Constantes
JSON_PATH = "data/encodings.json"
ANGULOS_REQUERIDOS = [
    "frontal",
    "perfil_derecho", 
    "perfil_izquierdo"
]

class SistemaIdentificacion:
    def __init__(self):
        self.known_data = self._cargar_datos()
        self.cap = None
        
    def _cargar_datos(self):
        """Carga los datos existentes del archivo JSON."""
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except Exception:
                    return []
        return []
    
    def _guardar_datos(self):
        """Guarda los datos en el archivo JSON."""
        os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.known_data, f, ensure_ascii=False, indent=2)
    
    def _calcular_angulo_facial(self, landmarks):
        """Calcula el ángulo facial basado en los landmarks."""
        if not landmarks or not all(k in landmarks for k in ['nose_bridge', 'left_eye', 'right_eye', 'top_lip']):
            return None
        
        # Obtener todos los puntos clave necesarios
        if not all(k in landmarks for k in ['nose_bridge', 'left_eye', 'right_eye', 'chin', 'nose_tip']):
            return None
            
        # Calcular puntos medios más estables
        ojo_izq = np.mean(landmarks['left_eye'], axis=0)
        ojo_der = np.mean(landmarks['right_eye'], axis=0)
        punto_medio_ojos = (ojo_izq + ojo_der) / 2
        
        # Usar el mentón y la punta de la nariz para el ángulo vertical
        menton = np.mean(landmarks['chin'], axis=0)
        nariz_punta = np.mean(landmarks['nose_tip'], axis=0)
        
        # Calcular ángulo horizontal usando múltiples puntos de referencia
        nariz_bridge = np.array(landmarks['nose_bridge'])
        vector_nariz = nariz_bridge[-1] - nariz_bridge[0]  # Vector desde la base a la punta del puente
        
        # Vector entre los ojos (de izquierda a derecha)
        vector_ojos = ojo_der - ojo_izq
        distancia_ojos = np.linalg.norm(vector_ojos)
        
        # Punto medio entre los ojos
        punto_medio_ojos = (ojo_izq + ojo_der) / 2
        
        # Calcular las distancias de los ojos a la nariz
        nariz_punto = nariz_bridge[0]
        dist_ojo_izq = np.linalg.norm(ojo_izq - nariz_punto)
        dist_ojo_der = np.linalg.norm(ojo_der - nariz_punto)
        
        # Calcular ratio de asimetría
        ratio_asimetria = (dist_ojo_der - dist_ojo_izq) / (dist_ojo_der + dist_ojo_izq)
        
        # Calcular vector desde el punto medio de los ojos hasta la nariz
        vector_central = nariz_punto - punto_medio_ojos
        
        # Calcular ángulo base usando el vector central
        angulo_base = degrees(atan2(vector_central[0], vector_central[1]))
        
        # Aplicar una función de suavizado para la zona frontal
        UMBRAL_FRONTAL = 0.15  # Ajusta este valor para hacer la zona frontal más o menos amplia
        if abs(ratio_asimetria) < UMBRAL_FRONTAL:
            # Dentro de la zona frontal, reducir el ángulo proporcionalmente
            factor = abs(ratio_asimetria) / UMBRAL_FRONTAL
            angulo_horizontal = angulo_base * factor
        else:
            # Fuera de la zona frontal, aplicar una transición suave
            signo = 1 if ratio_asimetria > 0 else -1
            factor = min(1.0, (abs(ratio_asimetria) - UMBRAL_FRONTAL) / (0.5 - UMBRAL_FRONTAL))
            angulo_horizontal = signo * (15 + factor * 30)  # Escala de 15° a 45°
        
        # Usar los ojos y la nariz para calcular la inclinación vertical
        if abs(angulo_horizontal) > 20:  # Si está de perfil
            # Para perfiles, usar distancia vertical entre ojos
            ojo_alto = min(ojo_izq[1], ojo_der[1])
            ojo_bajo = max(ojo_izq[1], ojo_der[1])
            diff_vertical = ojo_bajo - ojo_alto
            # Normalizar por la distancia horizontal entre ojos
            diff_horizontal = abs(ojo_der[0] - ojo_izq[0])
            if diff_horizontal > 0:
                angulo_vertical = degrees(atan2(diff_vertical, diff_horizontal))
            else:
                angulo_vertical = 0
        else:
            # Para vista frontal, usar el método anterior
            vector_vertical = punto_medio_ojos - menton
            vector_nariz = nariz_punta - menton
            vector_vertical = vector_vertical / np.linalg.norm(vector_vertical)
            vector_nariz = vector_nariz / np.linalg.norm(vector_nariz)
            angulo_vertical = degrees(np.arccos(np.clip(np.dot(vector_vertical, vector_nariz), -1.0, 1.0)))
            if nariz_punta[1] > punto_medio_ojos[1]:
                angulo_vertical = -angulo_vertical
            angulo_vertical = angulo_vertical * 0.5
        
        return angulo_horizontal, angulo_vertical
    
    def _validar_angulo_facial(self, landmarks, angulo_requerido):
        """Valida si el ángulo facial corresponde al requerido."""
        if not landmarks:
            return False
            
        angulos = self._calcular_angulo_facial(landmarks)
        if angulos is None:
            return False
            
        angulo_h, angulo_v = angulos
        
        # Definir rangos aceptables para cada ángulo
        rangos = {
            "frontal": ((-15, 15), (-10, 10)),  # (horizontal, vertical)
            "perfil_derecho": ((35, 55), (-15, 15)),  # Centrado en 45 grados
            "perfil_izquierdo": ((-55, -35), (-15, 15)) # Centrado en -45 grados
        }
        
        if angulo_requerido not in rangos:
            return False
            
        rango_h, rango_v = rangos[angulo_requerido]
        return (rango_h[0] <= angulo_h <= rango_h[1] and 
                rango_v[0] <= angulo_v <= rango_v[1])
    
    def _detectar_rostro(self, frame):
        """Detecta rostros en el frame y retorna sus ubicaciones y encodings."""
        # Voltear el frame horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
        return face_locations, face_encodings, face_landmarks, frame  # Devolver también el frame volteado
    
    def _dibujar_rostro(self, frame, face_location, label, color=(0, 255, 0), landmarks=None, angulo=None):
        """Dibuja un rectángulo y etiqueta alrededor del rostro."""
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Obtener dimensiones del frame
        height, width = frame.shape[:2]
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2

        # Dibujar guías de alineación si tenemos los landmarks
        if landmarks and angulo:
            if 'left_eye' in landmarks and 'right_eye' in landmarks:
                ojo_izq = np.mean(landmarks['left_eye'], axis=0).astype(int)
                ojo_der = np.mean(landmarks['right_eye'], axis=0).astype(int)
                
                if angulo == "perfil_derecho":
                    # Dibujar guía angular para el perfil derecho
                    guide_length = width // 4
                    guide_center_x = width // 2
                    guide_center_y = height // 2
                    
                    # Dibujar ángulo objetivo (45 grados)
                    target_angle = 45
                    dx = int(guide_length * np.cos(np.radians(target_angle)))
                    dy = int(guide_length * np.sin(np.radians(target_angle)))
                    
                    # Dibujar línea guía para el ángulo correcto
                    cv2.line(frame, (guide_center_x, guide_center_y), 
                            (guide_center_x + dx, guide_center_y), (0, 255, 255), 2)
                    
                    # Agregar información sobre los ángulos objetivo
                    cv2.putText(frame, "Perfil derecho - Objetivos:", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "H debe estar cerca de 45°", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "V debe estar cerca de 0°", (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Dibujar línea horizontal para mantener la cabeza nivelada
                    cv2.line(frame, (left-20, center_y), (right+20, center_y), (0, 255, 255), 1)
                    
                    # Mensajes de guía
                    cv2.putText(frame, "1. Gire la cabeza 45° (H ≈ 45°)", (10, height - 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, "2. Mantenga la cabeza nivelada (V ≈ 0°)", (10, height - 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                elif angulo == "perfil_izquierdo":
                    # Similar al perfil derecho pero con ángulo opuesto
                    guide_length = width // 4
                    guide_center_x = width // 2
                    guide_center_y = height // 2
                    target_angle = -45
                    dx = int(guide_length * np.cos(np.radians(target_angle)))
                    dy = int(guide_length * np.sin(np.radians(target_angle)))
                    cv2.line(frame, (guide_center_x, guide_center_y), 
                            (guide_center_x + dx, guide_center_y), (0, 255, 255), 2)
                
                # Dibujar círculo en el centro de la cara
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
        
        # Añadir la etiqueta con los valores de los ángulos
        cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame

    def _dibujar_guia_angulo(self, frame, angulo):
        """Dibuja una guía visual para ayudar al usuario a posicionar el rostro."""
        height, width = frame.shape[:2]
        
        guias = {
            "frontal": "Mire directamente a la cámara",
            "perfil_derecho": "Gire la cabeza hacia su derecha (45°)",
            "perfil_izquierdo": "Gire la cabeza hacia su izquierda (45°)"
        }
        
        # Dibujar texto de guía
        cv2.putText(frame, f"Capturando ángulo: {angulo}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, guias[angulo], (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def registrar_persona(self):
        """Registra una nueva persona con sus diferentes ángulos faciales y estado de lentes."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("No se pudo acceder a la cámara.")
            return False

        persona = {
            "rut": input("Ingrese RUT: "),
            "nombre": input("Ingrese nombre completo: "),
            "otros": input("Otros datos (opcional): "),
            "encodings": {}
        }

        usa_lentes = input("¿La persona usa lentes habitualmente? (s/n): ").strip().lower() == 's'
        persona["usa_lentes"] = usa_lentes

        for angulo in ANGULOS_REQUERIDOS:
            persona["encodings"][angulo] = {"sin_lentes": [], "con_lentes": []}
            estados = ["sin_lentes"]
            if usa_lentes:
                estados.append("con_lentes")
            for estado_lentes in estados:
                print(f"\nCapturando ángulo: {angulo} - Estado: {estado_lentes.replace('_', ' ')}")
                print("Siga las instrucciones en pantalla")
                print("Presione 'c' para capturar cuando el indicador esté en verde")
                print("Presione 'q' para cancelar")
                print(f"Asegúrese de estar {'usando' if estado_lentes=='con_lentes' else 'sin'} lentes antes de capturar.")

                encodings_capturados = []
                while len(encodings_capturados) < 1:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    face_locations, face_encodings, face_landmarks, frame = self._detectar_rostro(frame)
                    frame = self._dibujar_guia_angulo(frame, angulo)

                    for i, (face_location, landmarks) in enumerate(zip(face_locations, face_landmarks)):
                        angulos_actuales = self._calcular_angulo_facial(landmarks)
                        if angulos_actuales:
                            angulo_h, angulo_v = angulos_actuales
                            if angulo == "perfil_derecho":
                                h_diff = abs(angulo_h - 45)
                                v_diff = abs(angulo_v)
                                debug_info = f"H:{angulo_h:.1f}° (meta:45°) V:{angulo_v:.1f}° (meta:0°)"
                            else:
                                debug_info = f"H:{angulo_h:.1f}° V:{angulo_v:.1f}°"
                        else:
                            debug_info = "No se detectaron ángulos"

                        if self._validar_angulo_facial(landmarks, angulo):
                            color = (0, 255, 0)
                            label = f"¡Ángulo correcto! Presione 'c' | {debug_info}"
                        else:
                            color = (0, 0, 255)
                            label = f"Ajuste el ángulo | {debug_info}"

                        frame = self._dibujar_rostro(frame, face_location, label, color, landmarks, angulo)

                    cv2.imshow('Registro de Rostro', frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        self.cap.release()
                        cv2.destroyAllWindows()
                        return False

                    if key == ord('c') and face_encodings and face_landmarks:
                        if self._validar_angulo_facial(face_landmarks[0], angulo):
                            encodings_capturados.append(face_encodings[0].tolist())
                            print(f"Ángulo {angulo} ({estado_lentes}) capturado exitosamente!")
                        else:
                            print("Ángulo facial no válido, intente nuevamente")

                persona["encodings"][angulo][estado_lentes] = encodings_capturados

        self.known_data.append(persona)
        self._guardar_datos()
        print("\n¡Registro completado exitosamente!")
        self.cap.release()
        cv2.destroyAllWindows()
        return True

    def identificar_persona(self):
        """Identifica personas en tiempo real usando todos los ángulos registrados y el estado de lentes."""
        estado_lentes = input("¿Las personas a identificar están usando lentes? (s/n): ").strip().lower()
        estado_lentes = "con_lentes" if estado_lentes == "s" else "sin_lentes"

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("No se pudo acceder a la cámara.")
            return

        print("Presiona 'q' para salir")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            face_locations, face_encodings, face_landmarks, frame = self._detectar_rostro(frame)

            for i, (face_location, face_encoding, landmarks) in enumerate(zip(face_locations, face_encodings, face_landmarks)):
                nombre = "Desconocido"
                mejor_coincidencia = 1.0  # Umbral de distancia
                angulo_actual = None

                # Determinar el ángulo actual del rostro
                for angulo in ANGULOS_REQUERIDOS:
                    if self._validar_angulo_facial(landmarks, angulo):
                        angulo_actual = angulo
                        break

                # Comparar con todos los encodings conocidos, solo del estado de lentes correspondiente
                for persona in self.known_data:
                    for angulo, encodings_dict in persona["encodings"].items():
                        # Compatibilidad con registros antiguos (lista) y nuevos (dict)
                        if isinstance(encodings_dict, dict):
                            encodings = encodings_dict.get(estado_lentes, [])
                        else:
                            encodings = encodings_dict
                        for encoding in encodings:
                            distancia = face_recognition.face_distance([encoding], face_encoding)[0]
                            if distancia < mejor_coincidencia and distancia < 0.5:
                                mejor_coincidencia = distancia
                                nombre = f"{persona['nombre']}"

                color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)
                label = f"{nombre} ({angulo_actual if angulo_actual else 'ángulo no detectado'}, {estado_lentes.replace('_', ' ')})"
                frame = self._dibujar_rostro(frame, face_location, label, color)

            cv2.imshow('Reconocimiento en vivo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    sistema = SistemaIdentificacion()
    
    while True:
        print("\n=== Sistema de Identificación ===")
        print("1. Registrar nueva persona")
        print("2. Identificar personas")
        print("3. Salir")
        
        opcion = input("\nSeleccione una opción: ")
        
        if opcion == "1":
            sistema.registrar_persona()
        elif opcion == "2":
            sistema.identificar_persona()
        elif opcion == "3":
            break
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()