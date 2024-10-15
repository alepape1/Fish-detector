import os
import cv2
from threading import Thread
from motion_detection import detect_motion  # Asegúrate de que esta importación funcione correctamente
import tkinter as tk  # Asegúrate de importar tkinter
import os

class VideoProcessor:

    def __init__(self, folder_path, update_progress_callback=None, update_video_frame_callback=None, 
                 update_video_name_callback=None, completed_files_listbox=None, root=None):
                
        self.running = True  # Añade esta variable de control
        self.folder_path = folder_path
        self.update_progress = update_progress_callback
        self.update_video_frame = update_video_frame_callback
        self.update_video_name = update_video_name_callback
        self.completed_files_listbox = completed_files_listbox
        self.root = root  # Añadir la referencia a root

        # Crear la carpeta de salida si no existe
        if not os.path.exists("output"):
            os.makedirs("output")

    #Funcion para guardar el video resumen
    def save_video_segment(self,out, cap, start_time, end_time):


        # Asegurarse de que el start_time nunca sea menor que 0
        start_time = max(0, start_time - 1)  # Grabar 1 segundo antes
        end_time = end_time + 1  # Grabar 1 segundo después

        # Saltar al frame de inicio ajustado (1 segundo antes)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Obtener el tiempo actual en segundos
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            
            # Detener cuando el tiempo actual es mayor que el end_time ajustado (1 segundo después)
            if current_time > end_time:
                break
            
            out.write(frame)


    def process_video(self, video_path):
        
        print(f"The video path is: {video_path}")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"No se pudo abrir el video: {video_path}")
                return
         
            video_file_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"output/{video_file_name}_resumen.avi"

            # Si el archivo existe, eliminarlo antes de crear el nuevo segmento de video
            if os.path.exists(output_path):
                os.remove(output_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

            # Llama al callback para actualizar el nombre del video en la interfaz en el hilo de la GUI
            if self.update_video_name:
                self.root.after(0, self.update_video_name, f"Procesando: {video_file_name}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fish_detected = False  # Bandera para indicar si se ha detectado un pez
            
              # Initialize variables for this video
            fish_count = 0
            detected_fish = []
            start_time = None
            start_times = []
            end_times = []
            min_scene_duration = 1  # Minimum 1 seconds for a scene to be saved


            # Procesar cada fotograma
            for current_frame in range(total_frames):
           
                ret, frame = cap.read()
                if not ret:
                    print(f"Fin del video o error al leer el fotograma: {current_frame}")
                    break

                if not self.running:
                    print(f"Cancelado por el usuario: {current_frame}")
                    break

                # Lógica de detección de movimiento y conteo de peces aquí
                fish_counter = detect_motion(frame, detected_fish, fish_count)
                if fish_counter > 0 and start_time is None:
                    start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                elif fish_counter == 0 and start_time is not None:
                    end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    start_times.append(start_time)
                    end_times.append(end_time)
                    start_time = None
                    # Calcular y actualizar el progreso del video
                progress = round((current_frame / total_frames) * 100,1)
                # Mostrar el progreso en la consola
                print(f"Procesando {video_file_name}: {progress:.2f}% completado", end="\r")
                if self.update_progress:
                    self.root.after(0, self.update_progress, progress)
                # Actualizar el frame en la GUI si es necesario
                if self.update_video_frame:
                    self.root.after(0, self.update_video_frame, frame)


            # Guardar todos los segmentos en el video resumen
            for start_time, end_time in zip(start_times, end_times):
                
                # Add condition to check if the scene is longer than the minimum duration
                if end_time - start_time > min_scene_duration:
                    self.save_video_segment(out, cap, start_time, end_time)


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
                if not self.running:
                    break
                video_path = os.path.join(self.folder_path, video_file)
                self.process_video(video_path)

                if self.completed_files_listbox is not None:
                    self.root.after(0, self.completed_files_listbox.insert, tk.END, video_file)

        thread = Thread(target=process_thread)
        thread.start()


    def stop_processing(self):
        self.running = False  # Esto señalizará a los hilos que deben detenerse
