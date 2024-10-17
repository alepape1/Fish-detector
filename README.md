# Procesador de Videos para Conteo de Peces

## Descripción
Este proyecto es una aplicación de escritorio desarrollada en Python que procesa videos para detectar movimiento de peces, crustaceos o gelatinosos. Utiliza técnicas de visión por computadora con OpenCV para analizar el movimiento en los videos y generar resúmenes de los segmentos donde se detectan peces.

## Características
- Interfaz gráfica de usuario (GUI) para fácil interacción
- Selección de carpeta con videos para procesar
- Procesamiento en lote de múltiples videos
- Detección de movimiento para identificar peces
- Generación de videos resumen con los segmentos donde se detectan peces
- Visualización en tiempo real del procesamiento de video (opcional)
- Barra de progreso y contador de porcentaje completado

<img src="screenshots\image.png" alt="GUI de la app Fish-detector analizando un video donde un pez es detectado" width="600"/>



## Requisitos
- Python v3.12.2
- OpenCV v.4.9.0
- Tkinte
- Pillow (PIL)

## Instalación
1. Clona este repositorio:
```bash
git clone https://github.com/alepape1/Fish-detector.git
```
2. Instala las dependencias:
```bash
pip install opencv-python tkinter pillow
```
3. Ejecuta la aplicación:
```bash
python gui.py
```

## Uso
1. Ejecuta el script principal:
2. Usa el botón "Seleccionar carpeta" para elegir la carpeta con los videos a procesar.
3. Haz clic en "Procesar Videos" para iniciar el análisis.
4. Opcionalmente, puedes marcar o desmarcar "Mostrar video" para ver el procesamiento en tiempo real.
5. Los videos procesados se guardarán en la carpeta "output".

## Estructura del Proyecto
- `gui.py`: Contiene la interfaz gráfica de usuario.
- `video_processing.py`: Maneja el procesamiento de los videos.
- `motion_detection.py`: Implementa la lógica de detección de movimiento.

## Contribuir
Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de hacer un pull request.

## Licencia
[Incluir información sobre la licencia aquí]

## Contacto
asantana - asantanag@gmail.com  
Alejandro Santana Garcia Github:alepape1