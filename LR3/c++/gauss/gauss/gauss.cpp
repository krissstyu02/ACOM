#include <iostream>
#include <vector>
#include <cmath>
using namespace std;


// Реализация функции Гаусса
double gauss(double x, double  y, double  omega, double  a, double  b) {
    double omegaIn2 = 2 * pow(omega, 2);
    double m1 = 1 / (3.14 * omegaIn2);
    double m2 = exp(-(pow((x - a), 2) + pow((y - b), 2)) / omegaIn2);
    return m1 * m2;
}

// Реализация фильтра Гаусса
vector<vector<double>> gaussianBlur(vector<vector<double>>& img, int kernel_size, double standard_deviation) {
    int a = (kernel_size + 1) / 2;
    int b = (kernel_size + 1) / 2;

    // построение матрицы ядра свертки
    vector<vector<double>> kernel(kernel_size, vector<double>(kernel_size));
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            kernel[i][j] = gauss(i, j, standard_deviation, a, b);
        }
    }

    // нормализация матрицы ядра свертки
    double sum = 0;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            sum += kernel[i][j];
        }
    }
    cout << "Normilize matrix"<<endl;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            kernel[i][j] /= sum;
            cout << kernel[i][j] << " ";
        }
        cout << endl;
    }

    // применение операции свертки
    int img_height = img.size();
    int img_width = img[0].size();
    vector<vector<double>> img_blurred(img_height, vector<double>(img_width));
    for (int i = a; i < img_height - a; ++i) {
        for (int j = b; j < img_width - b; ++j) {
            double val = 0;
            for (int k = -a; k <= a; ++k) {
                for (int l = -b; l <= b; ++l) {
                    val += img[i + k][j + l] * kernel[k + a][l + b];
                }
            }
            img_blurred[i][j] = val;
        }
    }

    return img_blurred;
}

int main() {
    vector<vector<double>> img = { {1.0, 4.0, 7.0, 4.0, 1.0},
                                   {4.0, 16.0, 26.0, 16.0, 4.0},
                                   {7.0, 26.0, 41.0, 26.0, 7.0},
                                   {4.0, 16.0, 26.0, 16.0, 4.0},
                                   {1.0, 4.0, 7.0, 4.0, 1.0} };

    int kernel_size = 5;
    double standard_deviation = 3.0;

    cout << "matrix" << endl;
    // вывод размытого изображения
    for (const auto& row : img) {
        for (const auto& pixel : row) {
            cout << pixel << " ";
        }
        cout << endl;
    }

    vector<vector<double>> img_blurred = gaussianBlur(img, kernel_size, standard_deviation);
    cout << "Result matrix"<<endl;
    // вывод размытого изображения
    for (const auto& row : img_blurred) {
        for (const auto& pixel : row) {
            cout << pixel << " ";
        }
        cout << endl;
    }

    return 0;
}