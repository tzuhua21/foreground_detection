import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('video.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=100)


while(1):
	ret, frame = cap.read()
	frame=cv2.resize(frame,(480,360))
	
	ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
	channels = cv2.split(ycrcb)
	
	cv2.equalizeHist(channels[0], channels[0])
	cv2.merge(channels, ycrcb)
	cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, frame)


	fgmask = fgbg.apply(frame,learningRate = 0.002)
	
	smooth = cv2.GaussianBlur(fgmask,(5,5),cv2.BORDER_DEFAULT)
    
	th = cv2.threshold(smooth.copy(), 250, 255, cv2.THRESH_BINARY)[1]

	kernel = np.ones((5, 5), np.uint8)
	dilated = cv2.dilate(th, kernel)
	final = cv2.erode(dilated, kernel)
	#erosioned = cv2.erode(dilated, kernel)
	#final = cv2.dilate(erosioned, kernel)
   
	contours,hierarchy = cv2.findContours(final, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	cv2.imshow("foreground", final)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
