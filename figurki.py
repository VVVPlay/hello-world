import cv2 as cv

src = cv.imread('figures/1.jpg')
gr = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gr, 150, 250)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
contours = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
for cont in contours:
    sm = cv.arcLength(cont, True)
    apd = cv.approxPolyDP(cont, 0.02*sm, True)
    cv.drawContours(src, [apd], -1, (0, 255, 0), 4)
cv.imwrite('figures/result.jpg', src)
cv.imshow('rez', src)
cv.waitKey(0)
