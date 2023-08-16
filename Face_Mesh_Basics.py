# in this project we will learn how to detect 468 different landmarks on face
# We will use the model provide by google that runs in real time on CPU

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

cTime = 0
pTime = 0

mpDraw = mp.solutions.drawing_utils   # for drawing the mesh
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

while True:
    Success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_FACE_OVAL,
                                  drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                # print(id, lm)
                iw, ih, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id, x, y)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS:{int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == 80 or key == 113:
        break

cap.release()
cv2.destroyAllWindows()

print("Code Completed!")