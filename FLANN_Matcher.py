import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

MIN_MATCH_COUNT = 10

img1 = cv2.imread('./last_img0.jpg',0)          # queryImage
img2 = cv2.imread('./last_img69.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)
dst = None

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    h2, w2 = img2.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    pts2 = np.float32([ [0,0],[0,h2-1],[w2-1,h2-1],[w2-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts2,M)
    Mperspective = cv2.getPerspectiveTransform(dst, pts2)
    warpedImg = cv2.warpPerspective(img2,Mperspective,(w,h))

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.subplot(121),plt.imshow(warpedImg, 'gray'),plt.title('dst') 
plt.subplot(122),plt.imshow(img3, 'gray'),plt.title('output')
# plt.subplot(123),plt.imshow(img2, 'gray'),plt.title('img2')
plt.show()