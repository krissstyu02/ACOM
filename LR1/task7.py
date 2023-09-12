import cv2

# Функция для чтения видео с веб-камеры и записи его в файл
def readIPWriteTOFile():
    # Создание объекта для захвата видео с устройства (0 - индекс камеры по умолчанию)
    video = cv2.VideoCapture(0)

    # Получение ширины и высоты кадра
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Установка кодека для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Создание объекта для записи видео в файл "output_3.mov" с заданными параметрами
    video_writer = cv2.VideoWriter("output_3.mov", fourcc, 25, (w, h))

    # Бесконечный цикл для захвата и записи видео
    while (True):
        # Захват кадра с веб-камеры
        ok, img = video.read()

        # Отображение кадра на экране
        cv2.imshow('Webcam video', img)

        # Запись кадра в файл
        video_writer.write(img)

        # Выход из цикла при нажатии клавиши 'esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Освобождение ресурсов и закрытие окон
    video.release()
    cv2.destroyAllWindows()


readIPWriteTOFile()
