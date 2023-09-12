import cv2

# отображение видео в окне
cap =cv2.VideoCapture(r'C:\Disk CD\обои\фото\Новая папка\весна 2019\BDRD7996.MOV', cv2.WINDOW_NORMAL)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# изменение размера окна
# cv2.resizeWindow('Video', 800, 600)
cv2.resizeWindow('Video', 1024, 1000)
# cv2.resizeWindow('Video', 1800, 800)

# чтение видеофайла кадр за кадром
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # изменение цветовой гаммы кадра
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vsh = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # отображение кадра в окне
        cv2.imshow('Video',gray)

        # выход при нажатии клавиши 'esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
