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
    def ejecutar(self, datos):
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        while True:
            ret,frame = cap.read()
            if ret == False: break
            frame = cv2.flip(frame,1)

            face_locations = fr.face_locations(frame)
            if face_locations != []:
                for face in face_locations:
                    cv2.rectangle(frame,(face[3],face[0]),(face[1],face[2]),(0,255,0),2)

            cv2.imshow("Frame",frame)
            k = cv2.waitKey(1)
            if k == 27 & 0xFF:
                break

        cap.release()
        cv2.destroyAllWindows()