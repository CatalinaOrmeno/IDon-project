import cv2  # Captura de cámara
import face_recognition as fr  # Reconocimiento facial
import os
import json

# ------------------------------
# Clase base para todos los módulos
# ------------------------------
class ModuloPipeline:
    def __init__(self, siguiente_modulo=None):
        # Guardamos una referencia al siguiente módulo en la cadena
        self.siguiente_modulo = siguiente_modulo

    def procesar(self, datos):
        """
        Función principal para procesar datos.
        Llama al método 'ejecutar' del módulo actual y pasa el resultado al siguiente módulo.
        """
        # Ejecutar lógica propia del módulo
        resultado = self.ejecutar(datos)

        # Si hay resultado y hay un siguiente módulo, lo pasamos al siguiente
        if resultado is not None and self.siguiente_modulo:
            return self.siguiente_modulo.procesar(resultado)

        # Si no hay siguiente módulo o resultado es None, termina la cadena
        return resultado

    def ejecutar(self, datos):
        # Cada módulo debe implementar esta función
        raise NotImplementedError("Cada módulo debe implementar 'ejecutar'")


# ------------------------------
# Módulo 1: Captura de cámara
# ------------------------------
class CapturaCamara(ModuloPipeline):
    def __init__(self, siguiente_modulo=None):
        super().__init__(siguiente_modulo)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def ejecutar(self, datos=None):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)

    def liberar(self):
        self.cap.release()

# ------------------------------
# Módulo 2: Cargar encodings y nombres
# ------------------------------
class CargarEncodings(ModuloPipeline):
    def __init__(self, json_path, siguiente_modulo=None):
        super().__init__(siguiente_modulo)
        self.json_path = json_path
        self.known_encodings = []
        self.known_names = []
        self._cargar()

    def _cargar(self):
        if not os.path.exists(self.json_path):
            self.known_encodings = []
            self.known_names = []
            return
        with open(self.json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
        self.known_encodings = []
        self.known_names = []
        for p in data:
            if 'encodings' in p:
                for tipo in ['frontal', 'perfil_derecho', 'perfil_izquierdo']:
                    if tipo in p['encodings']:
                        encs = p['encodings'][tipo]
                        # Si es dict (nuevo formato)
                        if isinstance(encs, dict):
                            for estado in ['sin_lentes', 'con_lentes']:
                                for encoding in encs.get(estado, []):
                                    self.known_encodings.append(list(map(float, encoding)))
                                    self.known_names.append(p.get('nombre', 'Desconocido'))
                        # Si es lista (formato antiguo)
                        elif isinstance(encs, list):
                            for encoding in encs:
                                self.known_encodings.append(list(map(float, encoding)))
                                self.known_names.append(p.get('nombre', 'Desconocido'))

    def ejecutar(self, datos):
        # Pasa los encodings y nombres junto con el frame
        return {'frame': datos, 'encodings': self.known_encodings, 'names': self.known_names}

# ------------------------------
# Módulo 3: Detección de rostros
# ------------------------------
class DetectarRostros(ModuloPipeline):
    def ejecutar(self, datos):
        # datos puede ser frame o dict
        if isinstance(datos, dict):
            frame = datos['frame']
        else:
            frame = datos
        face_locations = fr.face_locations(frame)
        # Devuelve dict para el pipeline
        datos['frame'] = frame
        datos['face_locations'] = face_locations
        return datos

# ------------------------------
# Módulo 4: Calcular encodings de rostros detectados
# ------------------------------
class CalcularEncodings(ModuloPipeline):
    def ejecutar(self, datos):
        frame = datos['frame']
        face_locations = datos['face_locations']
        encodings = fr.face_encodings(frame, face_locations)
        datos['face_encodings'] = encodings
        return datos

# ------------------------------
# Módulo 5: Comparar y reconocer rostros
# ------------------------------
class CompararReconocimiento(ModuloPipeline):
    def ejecutar(self, datos):
        encodings = datos['face_encodings']
        known_encodings = datos['encodings']
        known_names = datos['names']
        nombres_detectados = []
        for encoding in encodings:
            matches = []
            if known_encodings:
                matches = fr.compare_faces(known_encodings, encoding, tolerance=0.45)
            if True in matches:
                name = known_names[matches.index(True)]
            else:
                name = "Desconocido"
            nombres_detectados.append(name)
        datos['nombres_detectados'] = nombres_detectados
        return datos

# ------------------------------
# Módulo 6: Dibujar rectángulos y nombres
# ------------------------------
class DibujarRectangulosConNombre(ModuloPipeline):
    def ejecutar(self, datos):
        frame = datos['frame']
        face_locations = datos['face_locations']
        nombres = datos.get('nombres_detectados', [])
        for i, (top, right, bottom, left) in enumerate(face_locations):
            w = right - left
            h = bottom - top
            aspect_ratio = w / h if h != 0 else 0
            # Solo dibujar si el rostro es frontal
            if 0.8 <= aspect_ratio <= 1.2:
                color = (0, 255, 0) if i < len(nombres) and nombres[i] != "Desconocido" else (0, 0, 255)
                label = nombres[i] if i < len(nombres) else ""
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

# ------------------------------
# Módulo 7: Mostrar en ventana
# ------------------------------
class MostrarVentana(ModuloPipeline):
    def ejecutar(self, frame):
        cv2.imshow("Camara", frame)
        return frame

# ------------------------------
# Orquestador principal
# ------------------------------
if __name__ == "__main__":
    # Armamos la cadena con reconocimiento facial
    pipeline = CapturaCamara(
        CargarEncodings("data/encodings.json",
            DetectarRostros(
                CalcularEncodings(
                    CompararReconocimiento(
                        DibujarRectangulosConNombre(
                            MostrarVentana()
                        )
                    )
                )
            )
        )
    )

    captura = pipeline  # alias para usar liberar() luego

    while True:
        frame = pipeline.procesar(None)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.liberar()
    cv2.destroyAllWindows()
