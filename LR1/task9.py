import cv2

# Создание объекта VideoCapture для подключения к IP-камере
#URL-адрес потока видео с IP-камеры
cap = cv2.VideoCapture("http://192.168.0.101:4747/video")

while True:
    # Считывание кадра с IP-камеры
    ret, frame = cap.read()

    if ret:
        # Отображение кадра с IP-камеры на экране
        cv2.imshow("Phone's camera", frame)

        # Ожидание нажатия клавиши 'q' для выхода из цикла
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        # Если возникла ошибка при чтении видео, выходим из цикла
        print("Ошибка чтения видео")
        break

# Освобождение ресурсов и закрытие окон
cap.release()
cv2.destroyAllWindows()

