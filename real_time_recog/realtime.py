import cv2 as cv
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np


clf, preproc = joblib.load('mnist_clf_svm.pkl')

def get_ict(img):
    # x, y, w, h = 0, 0, 300, 300

    # Change color-space from BGR -> Gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Gaussian Blur and Threshold
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    #blackhat = cv.morphologyEx(blur, cv.MORPH_BLACKHAT, kernel)
    _, thresh = cv.threshold(blur, 90, 255, cv.THRESH_BINARY_INV)
    #thresh = cv.dilate(blur, None)
    # Making thresh image of size x, y, w, h = 0, 0, 300, 300
    # thresh = thresh[y:y + h, x:x + w]


    # Find contours
    _, contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return img, contours, thresh


def main():

    # accessing camera
    capture = cv.VideoCapture(0)

    while(capture.isOpened()):
        ret, img = capture.read()

        img, contours, threshold = get_ict(img)

        result = ''

        if len(contours) > 0:
            contour = max(contours, key=cv.contourArea)
            if cv.contourArea(contour) > 1500 and cv.contourArea(contour) < 5000:
                x, y, w, h = cv.boundingRect(contour)

                newimg = threshold[y:y+h, x:x+w]

                newimg = cv.resize(newimg, (28, 28))
                newimg = np.array(newimg)

                hog_ft = hog(newimg, orientations=9, pixels_per_cell=(14, 14),
                             cells_per_block=(1, 1), block_norm='L2')

                hog_ft = preproc.transform(np.array([hog_ft], 'float64'))
                result = clf.predict(hog_ft)

        #constructing bounding rectangles
        x, y, w, h = 0, 0, 300, 300
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(img, "svm : " + str(result), (10, 320),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv.imshow("Frame", img)
        cv.imshow("Contours", threshold)
        k = cv.waitKey(10)
        if k == 27:
            break
    
main()
