import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox
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
        self.root.geometry("900x800")

        # Frame superior para los botones
        top_frame = tk.Frame(self.root, borderwidth=3, relief="groove", padx=10, pady=10)
        top_frame.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=30)

        # Botones en el frame superior
        self.select_button = tk.Button(top_frame, text="Seleccionar carpeta", command=self.select_folder)
        self.select_button.pack(side=tk.LEFT, padx=5)

        self.open_button = tk.Button(top_frame, text="Archivos procesados", command=self.open_output_folder)
        self.open_button.pack(side=tk.LEFT, padx=5)

        self.process_button = tk.Button(top_frame, text="Procesar Videos", state="disabled", command=self.process_videos)
        self.process_button.pack(side=tk.LEFT, padx=5)

     

        # Checkbox para mostrar video
        self.show_video_var = tk.BooleanVar(value=True)
        self.show_video_checkbox = tk.Checkbutton(top_frame, text="Mostrar video", variable=self.show_video_var)
        self.show_video_checkbox.pack(side=tk.LEFT, padx=5)

        # Frame inferior para el resto de elementos
        bottom_frame = tk.Frame(self.root, borderwidth=3, relief="groove", padx=10, pady=10)
        bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Label para mostrar el nombre del video
        self.video_name_label = tk.Label(bottom_frame, text="")
        self.video_name_label.pack(pady=5)

        # Label para mostrar el video
        self.video_label = tk.Label(bottom_frame)
        self.video_label.pack(pady=5)

        # Status label
        self.status_label = tk.Label(bottom_frame, text="Selecciona la carpeta de videos")
        self.status_label.pack(pady=10,padx=10)

       
        # Frame para la barra de progreso y el contador
        progress_frame = tk.Frame(bottom_frame)
        progress_frame.pack(fill=tk.X, pady=10, padx=20)

        # Botón para cerrar y salir
        self.close_button = tk.Button(progress_frame, text="Cerrar y Salir", command=self.close_and_quit)
        self.close_button.pack(side=tk.LEFT, padx=5)

        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Contador
        self.counter_var = tk.DoubleVar()
        self.counter_label = tk.Label(progress_frame, textvariable=self.counter_var , width=10)
        self.counter_label.pack(side=tk.LEFT, padx=(10, 0))  # Añade un poco de espacio a la izquierda

        self.folder_path = ""

    def close_and_quit(self):
        # Mostrar un cuadro de diálogo de confirmación
        if messagebox.askyesno("Confirmar salida", "¿Estás seguro de que quieres cerrar la aplicación y parar el proceso de analisis del video?"):
            if hasattr(self, 'processor'):
                self.processor.stop_processing()
                # Esperar un poco para que los hilos se detengan
                self.root.after(1000, self.force_quit)
            else:
                self.force_quit()
        # Si el usuario selecciona "No", simplemente retorna y no hace nada
    
    def force_quit(self):

        self.root.quit()
        self.root.destroy()

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
        self.counter_var.set(f"{int(progress)}%")
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
            self.processor = VideoProcessor(
                self.folder_path, 
                self.update_progress, 
                self.update_video_frame, 
                self.update_video_name,  # Pasar el callback para actualizar el nombre del video
                root=self.root  # Pasar root para usar after en el VideoProcessor
            )
            self.processor.process_videos()
            self.status_label.config(text="Procesando videos...")


if __name__ == "__main__":
    root = tk.Tk()
    gui = VideoGUI(root)
    root.mainloop()
