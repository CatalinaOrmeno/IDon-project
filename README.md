# IDon-project
---
# 🐍 Tutorial: Cómo usar el entorno virtual del proyecto

## 1. Crear y exportar el entorno (solo la primera vez)

1. Activa tu entorno actual:
   ```bash
   conda activate ENV311
   ```

2. Exporta todas las dependencias a un archivo:
    ```bash
    conda env export > environment.yml
    ```

Esto generará un archivo `environment.yml` en tu proyecto con todas las librerías y versiones exactas.

>📌 Consejo: si no quieres incluir rutas locales (como `prefix:`), abre `environment.yml` y elimina la última línea que empiece con `prefix:`.

## 2. Subir el archivo a GitHub

* Asegúrate de hacer commit y push del environment.yml junto con tu código.
* Ejemplo:

```bash
git add environment.yml
git commit -m "Agrego configuración del entorno"
git push origin feature/Cámara
```

## 3. Usar el entorno en otra máquina (tú o cualquier colaborador)

1. Clona el repositorio:
```bash
git clone https://github.com/TU_USUARIO/IDon-project.git
cd IDon-project
```

2. Crea el entorno con el archivo `environment.yml`:
```bash
conda env create -f environment.yml -n ENV311
```

👉 Esto instalará todas las librerías necesarias con sus versiones.

3. Activa el entorno:
```bash
conda activate ENV311
```
## 4. Verificación

Para comprobar que todo funciona, ejecuta:
```bash
python -c "import cv2, face_recognition, numpy; print('✅ Todo OK:', cv2.__version__, numpy.__version__)"
```
## 5. Actualizar el entorno (si agregas librerías nuevas)

Si en algún momento instalas más librerías, actualiza el archivo `environment.yml`:
```bash
conda env export > environment.yml
git add environment.yml
git commit -m "Actualizo dependencias"
git push
```