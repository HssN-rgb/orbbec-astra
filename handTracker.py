import cv2
import numpy as np
import handTrackingModule as htm
import time
import autopy
import monitorSelectionModule as msm

wCam, hCam = 640, 480
frameR = 100 #frame reduction
smoothing = 7
pTime = 0
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
screen = msm.get_monitor_by_index(0)
wScr , hScr = screen.width, screen.height


while True:
    #Finding the Landmark
    success, img = cap.read()
    img = detector.findHands(img)
    lmList,bbox = detector.findPosition(img)

    #Get the tip of index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:] # index
        x2, y2 = lmList[12][1:] # middle

        #Detect if the fingers are up
        fingers = detector.fingersUp()

        #Frame for your camera to use as a touchscreen
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        #Using Index finger to move
        if fingers[1] == 1 and fingers[2] == 0:

            #Convert Coordinates
            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1,(frameR,hCam-frameR),(0,hScr))

            #smoothing the shakes
            cLocX = pLocX + (x3-pLocX) / smoothing
            cLocY = pLocY + (y3-pLocY) / smoothing

            #Move Mouse
            autopy.mouse.move(wScr-cLocX,cLocY)
            cv2.circle(img,(x1,y1),10,(255,0,0),cv2.FILLED)
            pLocX, pLocY = cLocX, cLocY


        #using index and middle to click
        if fingers[1] == 1 and fingers[2] == 1:
            length,img, lineInfo= detector.findDistance(8,12,img)
            #if index and middle finger is close, mouse click
            if length < 30:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),10,(0,255,0),cv2.FILLED)
                autopy.mouse.click()


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv2.imshow('Virtual Mouse', img)
    cv2.waitKey(1)