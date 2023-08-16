import cv2
import mediapipe as mp
import time

class FaceMesh():
    def __init__(self, mode = False, maxFace = 2, refineLm = False,
                 detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.maxFace = maxFace
        self.refineLm = refineLm
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils  # for drawing the mesh
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxFace,
                                                 self.refineLm, self.detectionCon,
                                                 self.trackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    def drawFaceMesh(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        if self.results.multi_face_landmarks:
            for self.faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, self.faceLms, self.mpFaceMesh.FACEMESH_FACE_OVAL,
                                      self.drawSpec, self.drawSpec)
        return img

    def meshLandmarks(self, img):
        lmlist = []
        for id, lm in enumerate(self.faceLms.landmark):
            # print(id, lm)
            iw, ih, ic = img.shape
            x, y = int(lm.x * iw), int(lm.y * ih)
            lmlist.append([id, x, y])
        return lmlist

def main():

    cTime = 0
    pTime = 0

    cap = cv2.VideoCapture(0)
    Mesh = FaceMesh()
    while True:
        Success, img = cap.read()
        img = Mesh.drawFaceMesh(img)
        lm = Mesh.meshLandmarks(img)
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

if __name__ == "__main__":
    main()