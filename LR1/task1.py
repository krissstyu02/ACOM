import cv2

# Флаги для создания окна
cv2.namedWindow('Diswindow1',cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Diswindow2',cv2.WINDOW_NORMAL)
cv2.namedWindow('Diswindow3',cv2.WINDOW_FULLSCREEN)

# Флаги для чтения изображения
img1 = cv2.imread('logo2.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('logo.png', cv2.IMREAD_GRAYSCALE)  #серый
img3 = cv2.imread('logo 3.bmp', cv2.IMREAD_UNCHANGED)

cv2.imshow('Diswindow1', img1)
cv2.imshow('Diswindow2', img2)
cv2.imshow('Diswindow3', img3)
cv2.waitKey(0)


cv2.destroyAllWindows()


