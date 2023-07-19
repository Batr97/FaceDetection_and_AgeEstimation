import cv2 as cv
from retinaface import RetinaFace
import matplotlib.pyplot as plt


def detect_img(inImg):
    image = inImg.copy()
    resp = RetinaFace.detect_faces(image)
    faces = RetinaFace.extract_faces(image, align=True)
    # Draw a rectangle around the face
    if len(resp) > 0:
        for face in resp:
            y1, x1 = resp[face]['facial_area'][0], resp[face]['facial_area'][1]
            y2, x2 = resp[face]['facial_area'][2], resp[face]['facial_area'][3]
            cv.rectangle(inImg, (y1, x1), (y2, x2), (0, 255, 0), 3)
            cv.putText(inImg, face, (y1-3, x2-3), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    plt.imshow(inImg)

    return faces[0]


def detect_video(video):
    # Initialize variables to store the frame with the highest score
    max_score = 0
    best_frame = None
    # Read and process each frame
    for frame in video:
        faces = RetinaFace.detect_faces(frame)
        score = faces['face_1']['score']
        y1, x1, y2, x2 = faces['face_1']['facial_area']

        # Check if the current frame has a higher score than the previous best frame
        if score > max_score:
            max_score = score
            best_frame = frame.copy()
            y1_best, x1_best, y2_best, x2_best = y1, x1, y2, x2
            
    detected = best_frame[x1_best:x2_best, y1_best:y2_best]
    # Draw a rectangle around the face
    cv.rectangle(best_frame, (y1_best, x1_best), (y2_best, x2_best), (0, 255, 0), 2)
    plt.imshow(best_frame)
    return detected