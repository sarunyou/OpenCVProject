import glob
import cv2
import numpy as np

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image
paths = glob.glob('./last_img*')
templateImg = cv2.imread(paths[0], 1)
height, widht, channels = templateImg.shape 
print templateImg.shape
white = (255, 255, 255)
blankImage = create_blank(widht, height, rgb_color=white)
lenPaths = len(paths)
# test
# tempImage = cv2.imread(paths[0], 1);
# print 1.0/len(paths)
# blankImage = cv2.addWeighted(blankImage, 1.0 - 1.0/len(paths), tempImage, 1.0/len(paths), 0)


# temporary comment
for path in paths:
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(frame,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

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

        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        x, y, width, height = cv2.boundingRect(dst)
        # if x < 0: 
        #     # queryImageWidth += x
        #     x = 0
        # if y < 0:
        #     # queryImageHeight += y
        #     y = 0;

        try:
            roi = frame[y:y+queryImageHeight, x:x+queryImageWidth]
            copyImg1 = cv2.addWeighted(copyImg1, 0.5, roi, 0.5, 0)
            cv2.imshow('copyImg1', copyImg1)
            print 'x y width height is', x, y, width, height
            cv2.imshow('roi', roi)
            cv2.imwrite('roi%d.jpg' % count, roi)
            count+=1
        except :
            pass

        frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    tempImage = cv2.imread(path, 1);
    index = paths.index(path) + 1
    print 'index is', index
    alpha = 1.0 / index
    blankImage = cv2.addWeighted(blankImage, 1 - alpha, tempImage, alpha, 0)
    
cv2.imwrite('result.jpg', blankImage)
cv2.imshow('eiei', blankImage)
cv2.waitKey(0)
cv2.destoryAllWindows()

