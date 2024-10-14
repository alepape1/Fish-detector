import tkinter as tk
from tkinter import filedialog, ttk
from video_processing import VideoProcessor
import os
import cv2  # Importa OpenCV aquí
from PIL import Image, ImageTk  # Importar PIL para mostrar imágenes
import platform
import subprocess  # Necesario para abrir la carpeta en diferentes sistemas operativos


class VideoGUI:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Videos")
        self.root.geometry("800x600")

        # Progreso y estado
        self.progress_var = tk.DoubleVar()
        self.status_label = tk.Label(root, text="Selecciona la carpeta de videos")
        self.status_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(pady=10, padx=10, fill=tk.X)

        # Label para mostrar el video
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10, padx=10)

        # Label para mostrar el nombre del video
        self.video_name_label = tk.Label(root, text="")
        self.video_name_label.pack(pady=10)

        # Checkbox para mostrar video
        self.show_video_var = tk.BooleanVar(value=True)  # Valor inicial a True (mostrar video)
        self.show_video_checkbox = tk.Checkbutton(root, text="Mostrar video", variable=self.show_video_var)
        self.show_video_checkbox.pack(pady=10)

        # Botones
        self.select_button = tk.Button(root, text="Seleccionar carpeta", command=self.select_folder)
        self.select_button.pack(pady=10)

        self.open_button = tk.Button(root, text="Archivos procesados", command=self.open_output_folder)
        self.open_button.pack(pady=10)

        self.process_button = tk.Button(root, text="Procesar Videos", state="disabled", command=self.process_videos)
        self.process_button.pack(pady=10)

        self.folder_path = ""

    def open_output_folder(self):
        """Abrir la carpeta de salida en el explorador de archivos."""
        output_folder = "output"
        if os.path.exists(output_folder):
            if platform.system() == "Windows":
                os.startfile(output_folder)
            elif platform.system() == "Darwin":
                subprocess.call(["open", output_folder])
            else:
                subprocess.call(["xdg-open", output_folder])
        else:
            self.status_label.config(text="La carpeta de salida no existe.")

    def select_folder(self):
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.status_label.config(text=f"Carpeta seleccionada: {self.folder_path}")
            self.process_button.config(state="normal")

    def update_progress(self, progress):
        self.progress_var.set(progress)
        self.root.update_idletasks()

    def update_video_frame(self, frame):
        if frame is not None and self.show_video_var.get():  # Verifica si se debe mostrar el video
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)

            self.video_label.configure(image=image)
            self.video_label.image = image

    def update_video_name(self, video_name):
        """Actualizar el nombre del video actual en la interfaz."""
        self.video_name_label.config(text=video_name)

    def process_videos(self):
        if self.folder_path:
            processor = VideoProcessor(
                self.folder_path, 
                self.update_progress, 
                self.update_video_frame, 
                self.update_video_name,  # Pasar el callback para actualizar el nombre del video
                root=self.root  # Pasar root para usar after en el VideoProcessor
            )
            processor.process_videos()
            self.status_label.config(text="Procesando videos...")


if __name__ == "__main__":
    root = tk.Tk()
    gui = VideoGUI(root)
    root.mainloop()
