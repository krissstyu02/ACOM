from keras.models import load_model
from keras.utils import to_categorical
from keras.datasets import mnist
import time

# Подготовка данных
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)


#загрузка моделей
model_mlp = load_model(f'models/mlp_model25.keras')
model_cnn = load_model(f'models/cnn_model22.keras')

# Оцениваем производительность на тестовых данных
_,test_accuracy_mlp = model_mlp.evaluate(test_images, test_labels)
_,test_accuracy_cnn = model_cnn.evaluate(test_images, test_labels)

start_time = time.time()  # Засекаем начальное время обучения
# Предсказания модели на тестовых данных
predictions_mlp = model_mlp.predict(test_images)
end_time = time.time()  # Засекаем конечное время обучения
work_time_mlp=end_time-start_time

start_time = time.time()  # Засекаем начальное время обучения
# Предсказания модели на тестовых данных
predictions_cnn = model_cnn.predict(test_images)
end_time = time.time()  # Засекаем конечное время обучения
work_time_cnn=end_time-start_time

print(f"Модель_mlp: Процент корректной работы на тестовых данных: {round(test_accuracy_mlp*100,2)}%, "
          f" Скорость работы сети:{round(work_time_mlp,2)}")

print(f"Модель_cnn: Процент корректной работы на тестовых данных: {round(test_accuracy_cnn*100,2)}%, "
          f" Скорость работы сети:{round(work_time_cnn,2)}")
