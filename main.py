import os
os.environ['TF_CPP_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
import numpy as np

import cv2
import matplotlib.pyplot as plt

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


#NEURAL NETWORK WORKS   

(x_train, y_train), (x_test0, y_test) = mnist.load_data()
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test0.shape, y_test.shape))

AUTOGUESS=-1
x_train = x_train.reshape(AUTOGUESS, 28*28).astype("float32") / 255.0
x_test = x_test0.reshape(AUTOGUESS, 28*28).astype("float32") / 255.0
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'), 
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"]
)
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

inputImage = cv2.imread('phone_number.jpg')#your phone number

delta=2



# leng=img.shape[1]
# leng=int(leng/5)

# listl=[]

# for j in range(5):
#     listl.append(img[:,j*leng:(j+1)*leng-1])

# print(len(listl))

image_list=[]
by_sort=[]

def prepro(img):
    edges = cv2.Canny(img, 100, 150, apertureSize=5)
    lines = cv2.HoughLines(edges, 1, np.pi / 50, 50)
    print(lines)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 10)

    #cv2.imshow('marked', img)
    cv2.waitKey(0)
    cv2.imwrite('image.png', img)


    # 2 - remove horizontal lines

    img = cv2.imread("image.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_orig = cv2.imread("image.png")

    img = cv2.bitwise_not(img)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    #cv2.imshow("th2", th2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    horizontal = th2
    rows, cols = horizontal.shape

    # inverse the image, so that lines are black for masking
    horizontal_inv = cv2.bitwise_not(horizontal)
    # perform bitwise_and to mask the lines with provided mask
    masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
    # reverse the image back to normal
    masked_img_inv = cv2.bitwise_not(masked_img)
    #cv2.imshow("masked img", masked_img_inv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    horizontalsize = int(cols / 30)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    #cv2.imshow("horizontal", horizontal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # step1
    edges = cv2.adaptiveThreshold(horizontal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    #cv2.imshow("edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # step2
    kernel = np.ones((1, 2), dtype="uint8")
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    dilated = cv2.dilate(edges, kernel)
    #cv2.imshow("dilated", dilated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ctrs, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = img[y:y + h, x:x + w]

        # show ROI
        rect = cv2.rectangle(img_orig, (x, y), (x + w, y + h), (255, 255, 255), -1)

    #cv2.imshow('areas', rect)
    cv2.waitKey(0)

    cv2.imwrite('no_lines.png', rect)


    # 3 - detect and extract ROI's

    image = cv2.imread('no_lines.png')
    #cv2.imshow('i', image)
    cv2.waitKey(0)

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    cv2.waitKey(0)

    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('thresh', thresh)
    cv2.waitKey(0)

    # dilation
    kernel = np.ones((8, 45), np.uint8)  # values set for this image only - need to change for different images
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    #cv2.imshow('dilated', img_dilation)
    cv2.waitKey(0)

    # find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    delta=200
    for i, ctr in enumerate(sorted_ctrs):


        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y - delta:y + h + delta, x - delta:x + w + delta]

        # show ROI
        #cv2.imshow('segment no:'+str(i),roi)
        cv2.rectangle(image, (x - delta, y - delta), (x + w + delta, y + h + delta), (255, 255, 255), 1)
        #cv2.waitKey(0)

        # save only the ROI's which contain a valid information
        if h > 20 and w > 75:
            cv2.imwrite('roi\\{}.png'.format(i), roi)
            image_list.append(roi)
            #cv2.imshow('roi\\{}.png'.format(i), roi)

    cv2.imshow('marked areas', image)
    cv2.imwrite('m_areas.png', image)
    cv2.waitKey(0)

# for j in range(len(listl)):
#     prepro(listl[j])

def preprocess(img):
    inputImageCopy = img.copy()

    #BGR to grayscale
    grayscaleImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscaleImage=gammaCorrection(grayscaleImage,0.2)
    # Threshold via Otsu:
    threshValue, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Flood-fill border, seed at (0,0) and use black (0) color:
    cv2.floodFill(binaryImage, None, (0, 0), 0)
    # Get each bounding box
    # Find the big contours/blobs on the filtered image:
    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Look for the outer bounding boxes (no children):
    for _, c in enumerate(contours):

        # Get the bounding rectangle of the current contour:
        boundRect = cv2.boundingRect(c)

        # Get the bounding rectangle data:
        rectX = boundRect[0]
        rectY = boundRect[1]
        rectWidth = boundRect[2]
        rectHeight = boundRect[3]

        # Estimate the bounding rect area:
        rectArea = rectWidth * rectHeight
        #print(rectArea)
        # Set a min area threshold
        minArea = 1000 #depends on image
        maxArea = 10000
        # Filter blobs by area:
        if (rectArea > minArea) & (rectArea < maxArea):

            # Draw bounding box:
            color = (0, 255, 0)
            cv2.rectangle(inputImageCopy, (int(rectX), int(rectY)),
                          (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)
            #cv2.imshow("Bounding Boxes", inputImageCopy)

            # Crop bounding box:
            currentCrop = img[rectY-delta:rectY+rectHeight+delta,rectX-delta:rectX+rectWidth+delta]
            image_list.append(currentCrop)
            by_sort.append(rectX)
            #cv2.imshow("Current Crop", currentCrop)
            #cv2.waitKey(0)


preprocess(inputImage)
import math
#print(by_sort)

image_list=[x for _, x in sorted(zip(by_sort, image_list))]

for i in range(len(image_list)):
    cv2.imshow("Bounding Boxes", image_list[i])
    cv2.waitKey(0)



#print(image_list)

def transform_image(picture):
    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    picture=cv2.resize(picture,(50,100),interpolation=cv2.INTER_LINEAR)
    #plt.imshow(picture,cmap='gray')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    picture=cv2.erode(picture,kernel)
    #picture=gammaCorrection(picture,4)
    picture=cv2.bitwise_not(picture)
    picture=picture/255
    #print(picture)
    #print(picture)
    picture=cv2.copyMakeBorder(picture,40,40,40,40,cv2.BORDER_CONSTANT,value=0)
    picture=picture.reshape(picture.shape).astype('float32')
    picture=cv2.resize(picture,(28,28),interpolation=cv2.INTER_LINEAR)
    picture=np.where(picture<0.5,0,np.sqrt(picture))
    print(picture)
    plt.imshow(255*picture,cmap='gray')
    plt.show()
    return picture

img_list=[]
for j in range(len(image_list)):
    img_list.append(transform_image(image_list[j]))

for i in range(len(img_list)):
    verdict=model.predict(np.array([img_list[i].reshape((-1,1))]))
    print(np.argmax(verdict[0]))