import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from tkinter import Tk, filedialog
import logging

logging.basicConfig(filename="procesamiento.log", level=logging.INFO)

def log_info(mensaje):
    logging.info(mensaje)
    print(mensaje)

def detect_motion_gpu(frame, fgbg, kernel_gpu, stream):
    """
    Detecta movimiento en un frame utilizando CUDA.

    Args:
        frame (numpy.ndarray): Frame de entrada.
        fgbg (cv2.cuda_BackgroundSubtractorMOG2): Modelo de sustracción de fondo.
        kernel_gpu (cv2.cuda_GpuMat): Kernel para operaciones morfológicas.
        stream (cv2.cuda_Stream): Stream CUDA para procesamiento asíncrono.

    Returns:
        list: Lista de contornos detectados.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular el brillo medio
        mean_brightness = np.mean(gray)

        # Decidir si es noche o día
        es_noche = mean_brightness < 50  # Puedes ajustar este umbral
        
        # Mejora de contraste solo si es de día
        if not es_noche:
            gray = cv2.equalizeHist(gray)

        # Suavizado para reducir ruido sin perder bordes
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        mask = np.zeros_like(gray)
        h, w = gray.shape
        mask[40:h, :] = 255
        roi = cv2.bitwise_and(gray, gray, mask=mask)

        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(roi)

        fgmask_gpu = fgbg.apply(gpu_frame, learningRate=-1, stream=stream)
        stream.waitForCompletion()

        fgmask = fgmask_gpu.download()
        kernel = kernel_gpu.download()

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cnts
    except Exception as e:
        print(f"❌ Error en detect_motion_gpu: {e}")
        return []
    finally:
        if 'gpu_frame' in locals():
            gpu_frame.release()

def agrupar_frames(frames, max_salto=10):
    if not frames:
        return []
    frames = sorted(frames)
    grupos = [[frames[0]]]
    for f in frames[1:]:
        if f - grupos[-1][-1] <= max_salto:
            grupos[-1].append(f)
        else:
            grupos.append([f])
    return grupos

def procesar_video_resumen(video_path, mostrar=False, dibujar=False):
    print(f"Procesando: {video_path}")

    if not os.path.exists(video_path):
        print("❌ El archivo de video no existe.")
        return

    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        print("❌ CUDA no está disponible. Asegúrate de tener una GPU compatible y los drivers instalados.")
        return

    cap = cv2.VideoCapture(video_path)  # Crear el objeto VideoCapture
    if not cap.isOpened():
        print("❌ No se pudo abrir el archivo de video.")
        return

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception as e:
        print(f"❌ Error al obtener información del video: {e}")
        cap.release()  # Liberar el recurso en caso de error
        return

    print("📄 Información del video:")
    print(f"🧾 Resolución: {width}x{height}")
    print(f"🎞️ FPS: {fps}")
    print(f"📦 Total de frames: {total_frames}")

    fgbg = cv2.cuda.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_gpu = cv2.cuda_GpuMat()
    kernel_gpu.upload(kernel)

    motion_frames = set()
    salto = int(fps // 5)

    print("🔍 Detectando movimiento...")
    for i in tqdm(range(0, total_frames, salto)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        resize_factor = 0.5  # Factor de reducción
        resized = cv2.resize(frame, (int(width * resize_factor), int(height * resize_factor)))
        stream = cv2.cuda_Stream()
        cnts = detect_motion_gpu(resized, fgbg, kernel_gpu, stream)
        if any(cv2.contourArea(c) > 200 for c in cnts):
            for j in range(i - 10, i + 10 + 1):
                if 0 <= j < total_frames:
                    motion_frames.add(j)

    cap.release()  # Liberar el recurso después de procesar
    segmentos = agrupar_frames(motion_frames, max_salto=15)
    print(f"✅ Detectados {len(segmentos)} segmentos con movimiento.")

    # Crear carpeta para los resúmenes si no existe
    resumenes_dir = "resumenes"
    if not os.path.exists(resumenes_dir):
        os.makedirs(resumenes_dir)

    # Guardar el video de resumen en la carpeta
    resumen_filename = os.path.join(resumenes_dir, os.path.basename(video_path).replace(".mp4", "_resumen.mp4"))
    cap = cv2.VideoCapture(video_path)  # Reabrir el video para exportar el resumen
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(resumen_filename, fourcc, fps, (width, height))

    print("🎞️ Exportando video resumen...")
    for grupo in tqdm(segmentos):
        start = grupo[0]
        end = grupo[-1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(start, end + 1):
            ret, frame = cap.read()
            if not ret:
                break

            if dibujar:
                resize_factor = 0.5  # Factor de reducción
                resized = cv2.resize(frame, (int(width * resize_factor), int(height * resize_factor)))
                stream = cv2.cuda_Stream()
                cnts = detect_motion_gpu(resized, fgbg, kernel_gpu, stream)
                escala_x = width / (width // 2)
                escala_y = height / (height // 2)
                for c in cnts:
                    if 200 < cv2.contourArea(c) < 5000:  # Filtrar por área
                        x, y, w, h = cv2.boundingRect(c)
                        aspect_ratio = w / h
                        if 0.5 < aspect_ratio < 3.0:  # Filtrar por relación de aspecto
                            x, y, w, h = int(x * escala_x), int(y * escala_y), int(w * escala_x), int(h * escala_y)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            out.write(frame)

            if mostrar:
                cv2.imshow("Resumen", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    out.release()
    if mostrar:
        cv2.destroyAllWindows()

    print(f"🎬 Video resumen guardado en: {resumen_filename}")
    exportar_segmentos(video_path, segmentos)

def exportar_segmentos(video_path, segmentos):
    output_json = os.path.splitext(video_path)[0] + "_segmentos.json"
    data = {
        "video": video_path,
        "segmentos": [{"inicio": grupo[0], "fin": grupo[-1]} for grupo in segmentos],
    }
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)
    print(f"📄 Información de segmentos guardada en: {output_json}")

def procesar_carpeta(carpeta, mostrar=False, dibujar=False):
    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            procesar_video_resumen(os.path.join(carpeta, archivo), mostrar, dibujar)

def seleccionar_archivo():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(filetypes=[("Videos", "*.mp4;*.avi;*.mov;*.mkv")])

def seleccionar_carpeta():
    root = Tk()
    root.withdraw()
    return filedialog.askdirectory()

if __name__ == "__main__":
    resumenes_dir = "resumenes"
    if not os.path.exists(resumenes_dir):
        os.makedirs(resumenes_dir)

    print("Selecciona un archivo o carpeta:")
    opcion = input("1. Archivo\n2. Carpeta\n> ").strip()
    if opcion == "1":
        video_file = seleccionar_archivo()
        if video_file:
            procesar_video_resumen(video_file, mostrar=True, dibujar=True)
        else:
            print("❌ No se seleccionó ningún archivo.")
    elif opcion == "2":
        carpeta = seleccionar_carpeta()
        if carpeta:
            procesar_carpeta(carpeta, mostrar=True, dibujar=True)
        else:
            print("❌ No se seleccionó ninguna carpeta.")
    else:
        print("❌ Opción no válida.")
