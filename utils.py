import os
import cv2

def save_video_segment(out, cap, start_time, end_time):
    start_time = max(0, start_time - 1)
    end_time += 1
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    while True:
        ret, frame = cap.read()
        if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > end_time:
            break
        out.write(frame)

def write_file_txt(output_folder, video_file_name, start_times, end_times):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_name = os.path.join(output_folder, f"{video_file_name}_momentos_pesca.txt")
    with open(file_name, "w") as file:
        for start, end in zip(start_times, end_times):
            file.write(f"{start} - {end}\n")
