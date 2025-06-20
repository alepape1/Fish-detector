# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from multiprocessing import cpu_count, Pool, current_process
import traceback


# NOTA: cv2 y numpy se importarán dentro de funciones para intentar retrasar la carga,
# pero OpenCV bindings requiere numpy. Si numpy no se puede importar (paging file demasiado pequeño),
# se mostrará un mensaje claro.
def import_cv2_numpy():
    try:
        import numpy as np
    except Exception as e:
        print("Error al importar numpy:", e)
        print("Este error indica que numpy no pudo cargar sus extensiones C.")
        print("Habitualmente se debe a insuficiente memoria virtual (archivo de paginación) en Windows.")
        print("Por favor, aumenta el tamaño del archivo de paginación en Configuración de Sistema → Rendimiento → Memoria Virtual.")
        sys.exit(1)
    try:
        import cv2
    except Exception as e:
        print("Error al importar cv2:", e)
        print("Asegúrate de que OpenCV esté instalado correctamente en este entorno y las DLL estén accesibles.")
        print("También puede ayudar aumentar el archivo de paginación o reducir concurrencia.")
        sys.exit(1)
    return cv2, np

# ==== FUNCIONES DE UTILIDAD ==== 
def has_gui():
    """Comprueba si es posible usar GUI (tkinter)."""
    try:
        root = tk.Tk()
        root.withdraw()
        return True
    except:
        return False

def seconds_to_minutes_and_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    return int(minutes), int(seconds)

def select_folder():
    """Selecciona carpeta vía GUI si está disponible, sino usa ./video_files."""
    if has_gui():
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory()
        if folder:
            return folder
        else:
            return None
    else:
        # Por defecto, carpeta 'video_files' en el cwd
        default = os.path.join(os.getcwd(), "video_files")
        return default

# ==== DETECCIÓN DE MOVIMIENTO ==== 
def detect_motion(frame, detected_fish, fish_count, fgbg, kernel, kernel_gpu, use_cuda, min_contour_area=200, min_scene_duration=0.5):
    """
    Procesa un frame para detectar movimiento con background subtraction.
    Parámetros:
      - frame: imagen BGR (numpy array).
      - detected_fish: lista de bounding boxes previos (x,y,w,h) para evitar duplicados.
      - fish_count: contador acumulado de detecciones.
      - fgbg: objeto subtractor (CPU o GPU).
      - kernel: elemento estructurante para morfología (numpy array).
      - kernel_gpu: cv2.cuda_GpuMat con el kernel subido (o None si CPU).
      - use_cuda: bool, si usar GPU o no.
      - min_contour_area: área mínima para considerar un contorno como movimiento.
      - min_scene_duration: no se usa aquí, sino en el flujo principal.
    Retorna:
      - fish_count actualizado.
    """
    h, w = frame.shape[:2]
    # Dibujar rectángulo negro superior para texto
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    color = (0, 255, 0)
    texto_estado = "Estado: No se ha detectado movimiento"

    # Convertir a gris y enmascarar área de interés (por debajo de y=40)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    area_pts = np.array([[0, 40], [w, 40], [w, h], [0, h]])
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [area_pts], -1, 255, -1)
    image_area = cv2.bitwise_and(gray, gray, mask=mask)

    # Opcional: redimensionar para ahorrar GPU/RAM (descomentar si es necesario)
    # resize_factor = 0.5
    # image_area = cv2.resize(image_area, (0,0), fx=resize_factor, fy=resize_factor)

    # Background subtraction
    if use_cuda:
        try:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(image_area)
            # Pasar stream explícito (cv2.cuda.Stream.Null() si no se usa uno específico)
            fgmask_gpu = fgbg.apply(gpu_frame, None, cv2.cuda.Stream.Null())
            fgmask = fgmask_gpu.download()
        except cv2.error as e:
            # Si hay error GPU (por memoria), fallback a CPU
            print(f"[{current_process().pid}] Error GPU en apply MOG2: {e}. Reintentando en CPU.")
            fgmask = cv2.createBackgroundSubtractorMOG2().apply(image_area)
    else:
        fgmask = fgbg.apply(image_area)

    # Morfología en CPU (puede hacerse en GPU con funciones CUDA si se desea, pero aquí CPU)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    # Encontrar contornos
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > min_contour_area:
            x, y, w2, h2 = cv2.boundingRect(cnt)
            box = (x, y, w2, h2)
            if box not in detected_fish:
                # Nuevo movimiento detectado
                cv2.rectangle(frame, (x, y), (x + w2, y + h2), (0, 0, 255), 2)
                texto_estado = "Estado: ALERTA Movimiento Detectado!"
                color = (0, 0, 255)
                fish_count += 1
                detected_fish.append(box)
            else:
                # ya detectado antes: simplemente dibujar en otro color si se desea
                cv2.rectangle(frame, (x, y), (x + w2, y + h2), (0, 255, 0), 1)
    # Dibujar contorno global del área
    cv2.drawContours(frame, [area_pts], -1, color, 2)
    # Texto de estado y contador
    cv2.putText(frame, texto_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Peces detectados: {fish_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return fish_count

# ==== GUARDAR RESULTADOS ==== 
def write_file_txt(output_folder, video_file_name, start_times, end_times):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_path = os.path.join(output_folder, f"{video_file_name}_momentos_pesca.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        for i in range(len(start_times)):
            sm, ss = seconds_to_minutes_and_seconds(start_times[i])
            em, es = seconds_to_minutes_and_seconds(end_times[i])
            f.write(f"Pez {i + 1}: {sm:02d}m {ss:02d}s - {em:02d}m {es:02d}s\n")

def save_video_segment(out, cap, start_time, end_time):
    """
    Guarda un segmento de video entre start_time y end_time (en segundos).
    Ajusta 1 segundo antes y después para contexto.
    """
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

# ==== PROCESAR UN VIDEO ==== 
def process_video(video_path):
    """
    Procesa un video: detecta segmentos con movimiento (peces),
    extrae segmentos al video de resumen y escribe archivo txt con tiempos.
    Se inicializa subtractor y kernels aquí para cada proceso.
    """
    pid = current_process().pid
    print(f"[{pid}] Procesando: {video_path}")

    # Determinar uso de CUDA
    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if use_cuda:
        print(f"[{pid}] CUDA disponible. Se usará GPU.")
        try:
            fgbg = cv2.cuda.createBackgroundSubtractorMOG2()
        except cv2.error as e:
            print(f"[{pid}] Error al crear subtractor CUDA: {e}. Usando CPU en su lugar.")
            use_cuda = False
            fgbg = cv2.createBackgroundSubtractorMOG2()
    else:
        print(f"[{pid}] CUDA no disponible. Se usará CPU.")
        fgbg = cv2.createBackgroundSubtractorMOG2()

    # Kernel morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_gpu = None
    if use_cuda:
        try:
            kernel_gpu = cv2.cuda_GpuMat()
            kernel_gpu.upload(kernel)
        except cv2.error as e:
            print(f"[{pid}] Error al subir kernel a GPU: {e}. Continuando sin kernel GPU.")
            kernel_gpu = None

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{pid}] No se pudo abrir el video: {video_path}")
        return

    video_file_name = os.path.splitext(os.path.basename(video_path))[0]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Preparar carpeta de salida y VideoWriter para resumen
    output_folder = os.path.join(os.path.dirname(video_path), "output")
    os.makedirs(output_folder, exist_ok=True)
    resumen_path = os.path.join(output_folder, f"{video_file_name}_resumen.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(resumen_path, fourcc, fps, (frame_width, frame_height))

    # Variables de estado
    fish_count = 0
    detected_fish = []
    start_time_marker = None
    start_times, end_times = [], []
    min_scene_duration = 0.5  # segundos mínimos para considerar segmento

    # Procesar frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        # Mostrar progreso cada cierto número de frames
        if total_frames:
            progress = (frame_idx / total_frames) * 100
            print(f"[{pid}] {video_file_name}: {progress:.1f}% ", end="\r")
        else:
            print(f"[{pid}] {video_file_name}: frame {frame_idx}", end="\r")

        try:
            # Detectar movimiento
            prev_count = fish_count
            fish_count = detect_motion(frame, detected_fish, fish_count, fgbg, kernel, kernel_gpu, use_cuda)
        except Exception as e:
            print(f"\n[{pid}] Error en detect_motion: {e}")
            traceback.print_exc()
            break

        # Manejo de tiempos de inicio/fin de escena con movimiento
        if fish_count > 0 and start_time_marker is None:
            start_time_marker = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        elif fish_count == 0 and start_time_marker is not None:
            end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if end_time - start_time_marker >= min_scene_duration:
                start_times.append(start_time_marker)
                end_times.append(end_time)
            start_time_marker = None

    print(f"\n[{pid}] Finalizada lectura de frames para {video_file_name}. Extrayendo segmentos...")

    # Extraer segmentos
    # Reiniciar posición de captura
    for s, e in zip(start_times, end_times):
        try:
            save_video_segment(out, cap, s, e)
        except Exception as e:
            print(f"[{pid}] Error guardando segmento {s}-{e}: {e}")

    # Escribir archivo de texto con los tiempos
    try:
        write_file_txt(output_folder, video_file_name, start_times, end_times)
    except Exception as e:
        print(f"[{pid}] Error escribiendo archivo txt: {e}")

    out.release()
    cap.release()
    print(f"[{pid}] Procesamiento completado para {video_file_name}. Total peces detectados: {fish_count}")

def process_videos_in_folder_parallel(folder_path, max_workers=None):
    """
    Procesa todos los videos de la carpeta en paralelo.
    max_workers: número máximo de procesos simultáneos (por defecto cpu_count()).
    """
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        print("No se encontraron archivos de video en", folder_path)
        return
    video_paths = [os.path.join(folder_path, vf) for vf in video_files]
    n_cpus = cpu_count()
    workers = max_workers or n_cpus
    print(f"Lanzando procesamiento en paralelo con {workers} procesos (CPUs disponibles: {n_cpus})")
    with Pool(workers) as pool:
        pool.map(process_video, video_paths)

if __name__ == "__main__":
    print("OpenCV versión:", cv2.__version__)
    import_cv2_numpy()  # Importar cv2 y numpy para asegurar que están disponibles
    folder_path = select_folder()
    if not folder_path or not os.path.exists(folder_path):
        print("Carpeta no válida o no seleccionada:", folder_path)
    else:
        start = time.time()
        # Puedes ajustar max_workers a un número menor si tu máquina falla por memoria
        process_videos_in_folder_parallel(folder_path, max_workers=None)
        elapsed = time.time() - start
        print(f"Tiempo total de procesamiento: {elapsed:.2f} s")
