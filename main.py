import cv2
import time
import numpy as np
from multiprocessing import Pool
import os
import tkinter as tk
from tkinter import filedialog  # Asegúrate de incluir esta línea


# Inicialización de variables
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Función para detectar si hay un entorno gráfico
def has_gui():
    try:
        # Intenta crear una ventana oculta
        root = tk.Tk()
        root.withdraw()
        return True
    except Exception:
        return False

# Función para convertir segundos a minutos y segundos
def seconds_to_minutes_and_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    return int(minutes), int(seconds)

# Función para detectar movimiento en un fotograma
def detect_motion(frame, detected_fish, fish_count):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_height, frame_width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (frame_width, 40), (0, 0, 0), -1)
    color = (0, 255, 0)
    texto_estado = "Estado: No se ha detectado movimiento"
    
    area_pts = np.array([[0, 40], [frame_width, 40], [frame_width, frame_height], [0, frame_height]])
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)
    
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)
    
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        if cv2.contourArea(cnt) > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            if (x, y, w, h) not in detected_fish:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                texto_estado = "Estado: ALERTA Movimiento Detectado!"
                color = (0, 0, 255)
                fish_count += 1
                detected_fish.append((x, y, w, h))

    cv2.drawContours(frame, [area_pts], -1, color, 2)
    cv2.putText(frame, texto_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, "Peces detectados: " + str(fish_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return fish_count

# Función para escribir archivos de salida
def write_file_txt(output_folder, video_file_name, start_times, end_times):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    file_name = os.path.join(output_folder, f"{video_file_name}_momentos_pesca.txt")

    with open(file_name, "w") as file:
        for i in range(len(start_times)):
            start_minutes, start_seconds = seconds_to_minutes_and_seconds(start_times[i])
            end_minutes, end_seconds = seconds_to_minutes_and_seconds(end_times[i])
            file.write(f"Pez {i + 1}: {start_minutes:02d}m {start_seconds:02d}s - {end_minutes:02d}m {end_seconds:02d}s\n")

        # Si hay un pez detectado pero no finalizado, regístralo
        if len(start_times) > len(end_times):
            last_start_minutes, last_start_seconds = seconds_to_minutes_and_seconds(start_times[-1])
            file.write(f"Pez {len(start_times) + 1}: {last_start_minutes:02d}m {last_start_seconds:02d}s - {end_minutes:02d}m {end_seconds:02d}s\n")

# Función para procesar un video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video_file_name = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fish_count = 0
    start_time = None
    detected_fish = []
    start_times = []
    end_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress = (current_frame / total_frames) * 100
        print(f"Procesando {video_file_name}: {progress:.2f}% completado", end="\r")

        fish_counter = detect_motion(frame, detected_fish, fish_count)

        if fish_counter > 0 and start_time is None:
            start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        elif fish_counter == 0 and start_time is not None:
            end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            start_times.append(start_time)
            end_times.append(end_time)
            start_time = None

        k = cv2.waitKey(70) & 0xFF
        if k == 27:
            break

    write_file_txt("output", video_file_name, start_times, end_times)
    cap.release()

# Función para procesar videos en paralelo
def process_videos_in_folder_parallel(folder_path):
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4') or f.endswith('.avi') or f.endswith('.MOV')]
    video_paths = [os.path.join(folder_path, video_file) for video_file in video_files]

    with Pool() as pool:
        pool.map(process_video, video_paths)

# Verificar si hay GUI y seleccionar carpeta
def select_folder():
    if has_gui():
        root = tk.Tk()
        root.withdraw()  # Oculta la ventana principal de Tkinter
        folder_path = filedialog.askdirectory()
        return folder_path
    else:
        return os.path.join(os.getcwd(), "video_files")

if __name__ == "__main__":
    folder_path = select_folder()

    if not os.path.exists(folder_path):
        print("La carpeta no existe.")
    else:
        start_time = time.time()
        process_videos_in_folder_parallel(folder_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Tiempo total de procesamiento: {elapsed_time:.2f} segundos")
