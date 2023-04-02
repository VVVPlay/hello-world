import cv2

image = cv2.imread('2.jpg')
#размер
scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('a', image)
cv2.waitKey(0)
'''#обрезка
print(image.shape)
cropped = image[200:400, 200:400]
cv2.imshow('a', cropped)
cv2.waitKey(0)
#серый
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('a', gray_image)
cv2.waitKey(0)
#треш
ret, threshold_image = cv2.threshold(image, 127, 255, 0)
cv2.imshow('a', threshold_image)
cv2.waitKey(0)
#блюр
blurred = cv2.GaussianBlur(image, (51, 51), 0)
cv2.imshow('a', blurred)
cv2.waitKey(0)
#квадрат
output = image.copy()
cv2.rectangle(output, (200, 100), (300, 200), (0, 255, 55), 5)
cv2.imshow('a', output)
cv2.waitKey(0)
#линия
output = image.copy()
cv2.line(output, (60, 20), (400, 200), (0, 55, 255), 5)
cv2.imshow('a', output)
cv2.waitKey(0)
#текст
output = image.copy()
cv2.putText(output, "AAAAAAAAAAAA", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 4)
cv2.imshow('a', output)
cv2.waitKey(0)
'''


def filtor1(img):
    t, img = cv2.threshold(img, 155, 255, 0)
    img = cv2.GaussianBlur(img, (51, 1), 0)
    cv2.rectangle(img, (0, 0), (img.shape[0]-1, img.shape[1]-1), (0, 0, 0), 15)
    return img


def filtor2(img, n):
    for i in range(n):
        scale_percent = 33
        width = image.shape[1] * scale_percent // 100
        height = image.shape[0] * scale_percent // 100
        dim = (width, height)
        img1 = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img[img.shape[0]//3: img.shape[0]//3*2-2, img.shape[1]//3: img.shape[1]//3*2-2] = img1
    return img


filt = filtor1(image)
filt = filtor2(filt, 5)
cv2.imshow('a', filt)
cv2.waitKey(0)
