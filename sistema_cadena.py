import cv2 # Esta libreria se encarga de capturar la cámara.
import face_recognition as fr # Esta libreria se encarga de el proceso de reconocimiento facial.

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
# Módulo 2: Detección de rostros
# ------------------------------
class DetectarRostros(ModuloPipeline):
    def ejecutar(self, frame):
        face_locations = fr.face_locations(frame)
        return (frame, face_locations)

# ------------------------------
# Módulo 3: Dibujar rectángulos
# ------------------------------
class DibujarRectangulos(ModuloPipeline):
    def ejecutar(self, datos):
        frame, face_locations = datos
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        return frame

# ------------------------------
# Módulo 4: Mostrar en ventana
# ------------------------------
class MostrarVentana(ModuloPipeline):
    def ejecutar(self, frame):
        cv2.imshow("Camara", frame)
        return frame

# ------------------------------
# Orquestador principal
# ------------------------------
if __name__ == "__main__":
    # Armamos la cadena
    pipeline = CapturaCamara(
        DetectarRostros(
            DibujarRectangulos(
                MostrarVentana()
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
