import numpy as np
import cv2
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) # Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
old_gray = cv2.imread('./last_img0.jpg', 0) 
cv2.imshow('template', old_gray)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_gray)

frame_gray = translation_frame = cv2.imread('./last_img99.jpg', 0)
# calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
# Select good points
good_new = p1[st==1]
good_old = p0[st==1]
# draw the tracks
xSum = 0
ySum = 0
xArray = []
yArray = []
countRound = 0

for i,(new,old) in enumerate(zip(good_new,good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    xArray.append(c-a)
    yArray.append(d-b)
    countRound += 1
    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    frame = cv2.circle(frame_gray,(a,b),5,color[i].tolist(),-1)
print 'max xArray' , max(xArray)
print 'max yArray' , max(yArray)
# xArray.remove(max(xArray))
# xArray.remove(min(xArray))
# yArray.remove(max(yArray))
# yArray.remove(min(yArray))
xAvg = sum(xArray) / len(xArray)
yAvg = sum(yArray) / len(yArray)
print 'xArray', [x for x in xArray]
print 'yArray', [x for x in yArray]
print 'xAvg', xAvg
print 'yAvg', yAvg
rows, cols = frame.shape
print 'frame.shape', frame.shape
# transition frame by x equal xAvg, y equal yAvg
M = np.float32([[1, 0, xAvg], [0, 1, yAvg]])
dst = cv2.warpAffine(translation_frame, M, (cols, rows))
cv2.imshow('dst', dst)

img = cv2.add(frame,mask)
# img = cv2.addWeighted(old_gray, 0.3, frame_gray, 0.7, 0)
cv2.imshow('frame',img)
# Now update the previous frame and previous points
old_gray = frame_gray.copy()
p0 = good_new.reshape(-1,1,2)
cv2.waitKey(0)
cv2.destroyAllWindows()