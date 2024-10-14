import cv2
import numpy as np

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def detect_motion(frame, detected_fish, fish_count):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_height, frame_width = frame.shape[:2]
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
                fish_count += 1
                detected_fish.append((x, y, w, h))
                # Dibuja un rectángulo alrededor del pez detectado
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibuja el rectángulo
    
    # Muestra el frame procesado
    # cv2.imshow('Video Processing', frame)
    
    return fish_count
