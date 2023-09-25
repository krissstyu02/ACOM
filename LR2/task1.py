import cv2

# объект VideoCapture для подключения к IP-камере
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # преобразование изображения в формат HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("HSV_image", hsv)

    # сохранение изображения
    cv2.imwrite("task_1.png", hsv)

    # нажатие клавиши esc для выхода из цикла
    if cv2.waitKey(1) & 0xFF == 27:
        break

# освобождение ресурсов окна
cap.release()
cv2.destroyAllWindows()