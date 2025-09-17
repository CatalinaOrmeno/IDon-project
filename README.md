# IDon-project
---
# 🐍 Tutorial: Cómo usar el entorno virtual del proyecto

## 1. Usar el entorno en otra máquina (tú o cualquier colaborador)

1. Clona el repositorio:
```bash
git clone https://github.com/TU_USUARIO/IDon-project.git
cd IDon-project
```

2. Descarga Miniconda para Python 3.11 desde el instalador que esta en la carpeta `dependencies`.
   - Recuerda habilitar la opción de añadir al Path como variable de entorno.
   - Recuerda activar la opción de eliminar cache para evitar archivos basura.

### Crear y activar el entorno:
---
#### Opción 1 (con environment.yml):
3. Ubícate en la raíz de tu proyecto y ejecuta:
```powershell
conda env create -f dependencies/environment.yml
```

4. Activar el entorno (lo puedes hacer desde la consola o con la extensión de Python Environments):
```powershell
conda activate reconocimiento_facial
```
---
#### Opción 2 (Visual Studio Code):
3. Crea el entorno desde la interfaz de Visual Studio Code con la extención de Python Environments y asignaselo al proyecto.

4. Descarga librerias (Esto solo sirve como medida provicional) (Una vez que en la consola salga el env activado):
```powershell
# Instala librerías conda y pip de una sola vez
conda install -c conda-forge numpy matplotlib opencv dlib -y ; `
pip install face-recognition
```
---
## 2. Verificación

Para comprobar que todo funciona, ejecuta:
```bash
python -c "import cv2, face_recognition, numpy; print('✅ Todo OK:', cv2.__version__, numpy.__version__)"
```
## 3. Actualizar el entorno (si agregas o eliminas librerías)

Si en algún momento cambias las librerías, actualiza el archivo `environment.yml`:
```powershell
conda env update -f dependencies/environment.yml --prune
```