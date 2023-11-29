import cv2
import numpy as np
import time

# запуск видеопотока с камеры
cap = cv2.VideoCapture(0)
# cap =cv2.VideoCapture(r'test_video.mp4', cv2.WINDOW_NORMAL)
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)


# загрузка модели YOLOv3 для обнаружения лиц
net = cv2.dnn.readNetFromDarknet('YOLO/yolov3-face.cfg', 'YOLO/yolov3-wider_16000.weights')

# задание кодека и создание объекта VideoWriter для записи видео в файл
fourcc = cv2.VideoWriter_fourcc(*'XVID')
res = cv2.VideoWriter('yolo_result.avi', fourcc, 20.0, (640, 480))

# задание переменных для подсчета частоты потери изображения
frame_count = 0
prev_frame_time = 0

start_time = time.time()

# переменная для подсчета количества обнаруженных лиц
total_faces = 0

# чтение видеофайла кадр за кадром
while True:
    # чтение кадра из видеопотока
    ret, frame = cap.read()

    # подсчет количества кадров
    frame_count += 1

    # подсчет времени обработки кадра
    current_time = time.time()
    time_diff = current_time - prev_frame_time

    # получение списка обнаруженных лиц
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (415, 415), [0, 0, 0], True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # отрисовка прямоугольников вокруг лиц
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.98 and classId == 0:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)

    # подсчет количества обнаруженных лиц
    total_faces += len(outs)

    # вывод количества обнаруженных лиц на видео
    cv2.putText(frame, f'Total Faces: {total_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # запись кадра в выходной файл
    res.write(frame)

    # подсчет частоты потери изображения
    if time_diff > 1:
        fps = frame_count / time_diff
        print(f"Частота потери изображения: {1 / ((current_time - prev_frame_time) / frame_count):.0f} кадр(-a)(-ов)/секунду")
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
    print(f"Общее количество обнаруженных лиц: {total_faces}")

else:
    print("Видеофайл не содержит кадров.")

# освобождение ресурсов
cap.release()
res.release()
cv2.destroyAllWindows()
