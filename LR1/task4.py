import cv2

# отображение видео в окне
cap = cv2.VideoCapture(r'C:\Disk CD\обои\фото\Новая папка\весна 2019\BDRD7996.MOV',cv2.WINDOW_NORMAL)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# получение размеров кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# создание объект VideoWriter для записи видео в файл
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_1.mov', fourcc, 30.0, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        # отображение кадра в окне
        cv2.imshow('Video', frame)

        # выход при нажатии клавиши 'esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
