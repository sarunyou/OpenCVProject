import glob
import cv2
import numpy as np

def getTranslationImage(templat, image, numberImage):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) # Create some random colors
    color = np.random.randint(0,255,(100,3))
    p0 = cv2.goodFeaturesToTrack(templat, mask = None, **feature_params)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(templat, image, p0, None, **lk_params)
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
        # frame = cv2.circle(image,(a,b),5,color[i].tolist(),-1)
    xArray.remove(max(xArray))
    xArray.remove(min(xArray))
    yArray.remove(max(yArray))
    yArray.remove(min(yArray))
    xAvg = sum(xArray) / len(xArray)
    yAvg = sum(yArray) / len(yArray)
    rows, cols = image.shape
    print 'xAvg', xAvg
    print 'yAvg', yAvg
    # transition frame by x equal xAvg, y equal yAvg
    M = np.float32([[1, 0, xAvg], [0, 1, yAvg]])
    dst = cv2.warpAffine(image, M, (cols, rows))
    # cv2.imwrite('imageCalled%d.jpg' % numberImage, image)
    cv2.imwrite('translationImg%d.jpg' % numberImage, dst)
    print 'getTranslationImage'
    return dst

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
white = (255, 255, 255)
# test
# tempImage = cv2.imread(paths[0], 1);
# print 1.0/len(paths)
# blankImage = cv2.addWeighted(blankImage, 1.0 - 1.0/len(paths), tempImage, 1.0/len(paths), 0)

MIN_MATCH_COUNT = 25

# temporary comment
resultImg = cv2.imread(paths[0], 0)
templateSelectedImg = cv2.imread(paths[0], 0)

width, height = templateSelectedImg.shape
blankImage = create_blank(width, height, rgb_color=white)
lenPaths = len(paths)
countUseableImg = 0
sift = cv2.xfeatures2d.SIFT_create()
for path in paths[1:]:
    
    # Initiate SIFT detector
    cropImg = cv2.imread(path, 0)
    print path, paths.index(path)
    # cv2.imwrite('test%d.jpg' % paths.index(path), cropImg)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(templateSelectedImg,None)
    kp2, des2 = sift.detectAndCompute(cropImg,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good)>MIN_MATCH_COUNT:
            countUseableImg += 1
            print 'countUseableImg', countUseableImg
            cropImg = getTranslationImage(templateSelectedImg, cropImg, countUseableImg)
            alpha = 1.0 / countUseableImg
            resultImg = cv2.addWeighted(resultImg, 1 - alpha, cropImg, alpha, 0)
        else:
            pass
    except:
        pass

    
cv2.imshow('resultImg', resultImg)
cv2.imwrite('resultUseTransition.jpg', resultImg)
cv2.waitKey(0)
cv2.destoryAllWindows()

