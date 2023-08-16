import cv2
import time
import mediapipe as mp
import Face_Mesh_Module as fm

cap = cv2.VideoCapture(0)
detector = fm.FaceMesh()
cTime = 0
pTime = 0

while True:
    Success, img = cap.read()
    img = detector.drawFaceMesh(img)
    lm = detector.meshLandmarks(img)
    print(lm)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
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