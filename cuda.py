# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from multiprocessing import cpu_count, Pool

# ==== CONFIGURACIÓN GLOBAL ====
use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
if use_cuda:
    print("CUDA disponible. Se usará procesamiento en GPU.")
else:
    print("CUDA no disponible. Se usará procesamiento en CPU.")

# Inicialización de sustractor de fondo
if use_cuda:
    fgbg = cv2.cuda.createBackgroundSubtractorMOG2()
else:
    fgbg = cv2.createBackgroundSubtractorMOG2()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_gpu = cv2.cuda_GpuMat()
if use_cuda:
    kernel_gpu.upload(kernel)

# ==== FUNCIONES DE UTILIDAD ====
def has_gui():
    try:
        root = tk.Tk()
        root.withdraw()
        return True
    except:
        return False

def seconds_to_minutes_and_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    return int(minutes), int(seconds)

# ==== DETECCIÓN DE MOVIMIENTO ====
def detect_motion(frame, detected_fish, fish_count):
    frame_height, frame_width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (frame_width, 40), (0, 0, 0), -1)
    color = (0, 255, 0)
    texto_estado = "Estado: No se ha detectado movimiento"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    area_pts = np.array([[0, 40], [frame_width, 40], [frame_width, frame_height], [0, frame_height]])
    mask = np.zeros_like(gray)
    mask = cv2.drawContours(mask, [area_pts], -1, 255, -1)
    image_area = cv2.bitwise_and(gray, gray, mask=mask)

    if use_cuda:
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(image_area)
        fgmask_gpu = fgbg.apply(gpu_frame)
        fgmask = fgmask_gpu.download()
    else:
        fgmask = fgbg.apply(image_area)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        if cv2.contourArea(cnt) > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            if (x, y, w, h) not in detected_fish:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                texto_estado = "Estado: ALERTA Movimiento Detectado!"
                color = (0, 0, 255)
                fish_count += 1
                detected_fish.append((x, y, w, h))

    cv2.drawContours(frame, [area_pts], -1, color, 2)
    cv2.putText(frame, texto_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, "Peces detectados: " + str(fish_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return fish_count

# ==== ESCRITURA DE ARCHIVO DE RESULTADOS ====
def write_file_txt(output_folder, video_file_name, start_times, end_times):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = os.path.join(output_folder, f"{video_file_name}_momentos_pesca.txt")
    with open(file_path, "w") as f:
        for i in range(len(start_times)):
            sm, ss = seconds_to_minutes_and_seconds(start_times[i])
            em, es = seconds_to_minutes_and_seconds(end_times[i])
            f.write(f"Pez {i + 1}: {sm:02d}m {ss:02d}s - {em:02d}m {es:02d}s\n")

# ==== GUARDAR SEGMENTOS DE VIDEO ====
def save_video_segment(out, cap, start_time, end_time):
    start_time = max(0, start_time - 1)
    end_time += 1
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if current_time > end_time:
            break
        out.write(frame)

# ==== PROCESAR VIDEO ====
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video_file_name = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    out = cv2.VideoWriter(os.path.join(output_folder, f"{video_file_name}_resumen.avi"), cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    fish_count = 0
    start_time_marker = None
    detected_fish = []
    start_times, end_times = [], []
    min_scene_duration = 0.5  # en segundos

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress = (current_frame / total_frames) * 100
        print(f"Procesando {video_file_name}: {progress:.2f}%", end="\r")

        current_fish = detect_motion(frame, detected_fish, fish_count)

        if current_fish > 0 and start_time_marker is None:
            start_time_marker = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        elif current_fish == 0 and start_time_marker is not None:
            end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if end_time - start_time_marker >= min_scene_duration:
                start_times.append(start_time_marker)
                end_times.append(end_time)
            start_time_marker = None

    for s, e in zip(start_times, end_times):
        save_video_segment(out, cap, s, e)

    write_file_txt(output_folder, video_file_name, start_times, end_times)
    out.release()
    cap.release()

# ==== PROCESAMIENTO PARA TODA LA CARPETA ====
def process_videos_in_folder_parallel(folder_path):
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    video_paths = [os.path.join(folder_path, vf) for vf in video_files]
    with Pool(cpu_count()) as pool:
        pool.map(process_video, video_paths)

# ==== SELECCIÓN DE CARPETA ====
def select_folder():
    if has_gui():
        root = tk.Tk()
        root.withdraw()
        return filedialog.askdirectory()
    else:
        return os.path.join(os.getcwd(), "video_files")

if __name__ == "__main__":
    print("OpenCV versión:", cv2.__version__)
    folder_path = select_folder()
    if os.path.exists(folder_path):
        start = time.time()
        process_videos_in_folder_parallel(folder_path)
        print(f"Tiempo total: {time.time() - start:.2f} s")
    else:
        print("Carpeta no válida.")
