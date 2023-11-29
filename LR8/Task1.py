import cv2
import time

# запуск видеопотока с камеры
cap = cv2.VideoCapture(0)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# загрузка каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalface_default.xml')

# задание кодека и создание объекта VideoWriter для записи видео в файл
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('haarscade_output_1.avi', fourcc, 20.0, (640, 480))

# задание переменных для подсчета частоты потери изображения
frame_count = 0
prev_frame_time = 0

start_time = time.time()

# чтение видеофайла кадр за кадром
while True:
    # чтение кадра из видеопотока
    ret, frame = cap.read()

    if ret:

        # подсчет количества кадров
        frame_count += 1

        # подсчет времени обработки кадра
        current_time = time.time()
        time_diff = current_time - prev_frame_time

        # преобразование кадра в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # обнаружение лиц на кадре
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # отрисовка прямоугольников вокруг лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # вывод количества обнаруженных лиц на видео
        cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # запись кадра в выходной файл
        out.write(frame)

        # подсчет частоты потери изображения
        if time_diff > 1:
            fps = frame_count / time_diff
            print(
                f"Частота потери изображения: {1 / ((current_time - prev_frame_time) / frame_count):.0f} кадр(-a)(-ов)/секунду")
            prev_frame_time = current_time
            frame_count = 0

        # отображение кадра с обнаруженными лицами
        cv2.imshow('Video', frame)

        # выход при нажатии клавиши 'esc'
        if cv2.waitKey(1) & 0xFF == 27:
            break

end_time = time.time()

# вывод сравнительных характеристик
if cap.get(cv2.CAP_PROP_FRAME_COUNT) != 0:
    print(f"Время работы метода: {end_time - start_time:.5f} секунд")
    print(f"Скорость обработки: {cap.get(cv2.CAP_PROP_FPS):.0f} кадр(-a)(-ов)/секунду")
else:
    print("Видеофайл не содержит кадров.")

# освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()
