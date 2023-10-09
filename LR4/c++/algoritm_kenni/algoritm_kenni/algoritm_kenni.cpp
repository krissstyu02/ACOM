#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void kenni(const Mat& image, int kernel_size, int standart_deviation, float high_threshold, float low_threshold) {
    // Применение размытия по Гауссу
    Mat blurred_image;
    GaussianBlur(image, blurred_image, Size(kernel_size, kernel_size), standart_deviation);


    // Вывод черно-белого изображения после размытия
    imshow("Blurred Image", blurred_image);
    waitKey(0);

    // Вычисление градиентов с помощью функции Sobel
    Mat gradientX, gradientY;
    Sobel(blurred_image, gradientX, CV_32F, 1, 0); //CV_32F: Это тип данных для выходной матрицы gradientX.
                                                    //В данном случае, это 32-битные числа с плавающей точкой.
                                                    //1: Это порядок производной, который указывает, что мы хотим вычислить первую производную 
                                                        //градиента по оси X
    Sobel(blurred_image, gradientY, CV_32F, 0, 1);

    // Вычисление значений длин и углов градиентов
    Mat magnitude, angle;
    cartToPolar(gradientX, gradientY, magnitude, angle, true);

    // Вывод матриц значений длин и углов градиентов
    imshow("Gradient Magnitude", magnitude);
    imshow("Gradient Angle", angle);
    waitKey(0);

    // Подавление немаксимумов

// Создаем матрицу для хранения результатов подавления немаксимумов.
// Инициализируем ее нулями и устанавливаем такой же размер, как исходное изображение.
    Mat suppressed = Mat::zeros(image.size(), CV_32F);

    // Проходим по пикселям изображения, исключая граничные пиксели (без них невозможно выполнить операцию)
    for (int y = 1; y < image.rows - 1; ++y) {
        for (int x = 1; x < image.cols - 1; ++x) {
            // Получаем значение угла градиента в текущем пикселе
            float angleVal = angle.at<float>(y, x);

            // Проверяем, принадлежит ли угол к группе 0-45 градусов или 135-180 градусов
            if ((angleVal >= 0 && angleVal <= 45) || (angleVal >= 135 && angleVal <= 180)) {
                // Если угол принадлежит к одной из групп, сравниваем значение текущего пикселя
                // с соседними пикселями в направлении градиента (горизонтально)
                if (magnitude.at<float>(y, x) >= magnitude.at<float>(y, x - 1) &&
                    magnitude.at<float>(y, x) >= magnitude.at<float>(y, x + 1)) {
                    // Если текущий пиксель имеет максимальное значение в направлении градиента, сохраняем его значение
                    suppressed.at<float>(y, x) = magnitude.at<float>(y, x);
                }
            }
            else {
                // Если угол принадлежит к другой группе, сравниваем значение текущего пикселя
                // с соседними пикселями в направлении градиента (вертикально)
                if (magnitude.at<float>(y, x) >= magnitude.at<float>(y - 1, x) &&
                    magnitude.at<float>(y, x) >= magnitude.at<float>(y + 1, x)) {
                    // Если текущий пиксель имеет максимальное значение в направлении градиента, сохраняем его значение
                    suppressed.at<float>(y, x) = magnitude.at<float>(y, x);
                }
            }
        }
    }

    // Вывод изображения после подавления немаксимумов
    imshow("Suppressed Image", suppressed);
    waitKey(0);

    // Двойная пороговая фильтрация

// Создаем матрицу edges для хранения результатов двойной пороговой фильтрации.
// Инициализируем ее нулями и устанавливаем такой же размер, как исходное изображение.
    Mat edges = Mat::zeros(image.size(), CV_8U);

    // Проходим по всем пикселям исходного изображения
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            // Получаем значение градиента после подавления немаксимумов для текущего пикселя
            float val = suppressed.at<float>(y, x);

            // Сравниваем значение с высоким порогом для определения сильных границ
            if (val >= high_threshold) {
                // Если значение градиента выше или равно высокому порогу, устанавливаем пиксель в белый цвет (255)
                edges.at<uchar>(y, x) = 255;
            }
            // Если значение градиента между высоким и низким порогами, считаем его слабой границей
            else if (val >= low_threshold) {
                // Если значение градиента находится между высоким и низким порогами, устанавливаем пиксель в серый цвет (127)
                edges.at<uchar>(y, x) = 127;
            }
            // Если значение градиента ниже низкого порога, считаем его фоном и оставляем пиксель черным (0)
        }
    }


    // Вывод изображения после двойной пороговой фильтрации
    imshow("Edges", edges);
    waitKey(0);
}

int main() {
    string image_path = "C:/Users/Кристина/PycharmProjects/ACIOM/LR4/test4.jpg"; // Задайте путь к изображению
    int kernel_size = 5;        // Размер ядра для размытия по Гауссу
    int standart_devalvation = 50;        // стандартное отклонение
    float high_threshold = 100; // Высокий порог для пороговой фильтрации
    float low_threshold = 50;   // Низкий порог для пороговой фильтрации

    Mat image = imread(image_path, IMREAD_GRAYSCALE);

    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    kenni(image, kernel_size,standart_devalvation, high_threshold, low_threshold);

    return 0;
}
