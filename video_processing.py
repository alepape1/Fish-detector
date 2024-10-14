import os
import cv2
from threading import Thread
from motion_detection import detect_motion  # Asegúrate de que esta importación funcione correctamente
from utils import save_video_segment, write_file_txt  # Asegúrate de que estas funciones existan
import tkinter as tk  # Asegúrate de importar tkinter


class VideoProcessor:

    def __init__(self, folder_path, update_progress_callback=None, update_video_frame_callback=None, 
                 update_video_name_callback=None, completed_files_listbox=None, root=None):
        self.folder_path = folder_path
        self.update_progress = update_progress_callback
        self.update_video_frame = update_video_frame_callback
        self.update_video_name = update_video_name_callback
        self.completed_files_listbox = completed_files_listbox
        self.root = root  # Añadir la referencia a root

        # Crear la carpeta de salida si no existe
        if not os.path.exists("output"):
            os.makedirs("output")

    def process_video(self, video_path):
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"No se pudo abrir el video: {video_path}")
                return

            video_file_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"output/{video_file_name}_resumen.avi"
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

            # Llama al callback para actualizar el nombre del video en la interfaz en el hilo de la GUI
            if self.update_video_name:
                self.root.after(0, self.update_video_name, f"Procesando: {video_file_name}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fish_detected = False  # Bandera para indicar si se ha detectado un pez

            # Procesar cada fotograma
            for current_frame in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    print(f"Fin del video o error al leer el fotograma: {current_frame}")
                    break

                fish_count = detect_motion(frame, [], 0)

                # Calcular y actualizar el progreso del video
                progress = (current_frame / total_frames) * 100
                if self.update_progress:
                    self.root.after(0, self.update_progress, progress)

                # Guardar el frame en el video de salida
                out.write(frame)  # Guarda todos los frames para mantener la velocidad original

                # Actualizar el frame en la GUI si es necesario
                if self.update_video_frame:
                    self.root.after(0, self.update_video_frame, frame)

                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()  # Asegúrate de liberar el VideoWriter
            cv2.destroyAllWindows()

            if fish_detected:
                print(f"Peces detectados en el video: {video_file_name}, video guardado en: {output_path}")
            else:
                print(f"No se detectaron peces en el video: {video_file_name}")

        except Exception as e:
            print(f"Ocurrió un error al procesar el video {video_path}: {str(e)}")


    def process_videos(self):
        video_files = [f for f in os.listdir(self.folder_path) if f.endswith(('.mp4', '.avi', '.MOV'))]

        def process_thread():
            for i, video_file in enumerate(video_files, 1):
                video_path = os.path.join(self.folder_path, video_file)
                self.process_video(video_path)

                if self.completed_files_listbox is not None:
                    self.root.after(0, self.completed_files_listbox.insert, tk.END, video_file)

        thread = Thread(target=process_thread)
        thread.start()
