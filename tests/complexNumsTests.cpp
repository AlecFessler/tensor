#include "complexNum.h"
#include <iostream>

// test all 4 arithmetic operations, with float and double, and with complex and real numbers in both sides of operand since it's different for each operation
// also test conjugate, absolute value and exponential functions

int main() {
    Complex<float> a(1, 2);
    Complex<float> b(3, 4);

    Complex<double> c(1, 2);
    Complex<double> d(3, 4);

    float e = 5;
    double f = 5;

    // addition
    Complex<float> sum1 = a + b;
    Complex<float> sum2 = e + b;
    Complex<float> sum3 = a + e;

    Complex<double> sum4 = c + d;
    Complex<double> sum5 = f + d;
    Complex<double> sum6 = c + f;

    std::cout << sum1 << std::endl;
    std::cout << sum2 << std::endl;
    std::cout << sum3 << std::endl;

    std::cout << sum4 << std::endl;
    std::cout << sum5 << std::endl;
    std::cout << sum6 << std::endl;

    // subtraction
    Complex<float> sub1 = a - b;
    Complex<float> sub2 = e - b;
    Complex<float> sub3 = a - e;

    Complex<double> sub4 = c - d;
    Complex<double> sub5 = f - d;
    Complex<double> sub6 = c - f;

    std::cout << sub1 << std::endl;
    std::cout << sub2 << std::endl;
    std::cout << sub3 << std::endl;

    std::cout << sub4 << std::endl;
    std::cout << sub5 << std::endl;
    std::cout << sub6 << std::endl;

    // multiplication
    Complex<float> mul1 = a * b;
    Complex<float> mul2 = e * b;
    Complex<float> mul3 = a * e;

    Complex<double> mul4 = c * d;
    Complex<double> mul5 = f * d;
    Complex<double> mul6 = c * f;

    std::cout << mul1 << std::endl;
    std::cout << mul2 << std::endl;
    std::cout << mul3 << std::endl;

    std::cout << mul4 << std::endl;
    std::cout << mul5 << std::endl;
    std::cout << mul6 << std::endl;

    // division
    Complex<float> div1 = a / b;
    Complex<float> div2 = e / b;
    Complex<float> div3 = a / e;

    Complex<double> div4 = c / d;
    Complex<double> div5 = f / d;
    Complex<double> div6 = c / f;

    std::cout << div1 << std::endl;
    std::cout << div2 << std::endl;
    std::cout << div3 << std::endl;

    std::cout << div4 << std::endl;
    std::cout << div5 << std::endl;
    std::cout << div6 << std::endl;

    // conjugate
    Complex<float> conj1 = a.conj();
    Complex<double> conj2 = c.conj();

    std::cout << conj1 << std::endl;
    std::cout << conj2 << std::endl;

    // absolute value
    float abs1 = a.abs();
    double abs2 = c.abs();

    std::cout << abs1 << std::endl;
    std::cout << abs2 << std::endl;

    // exponential
    Complex<float> exp1 = a.exp();
    Complex<double> exp2 = c.exp();

    std::cout << exp1 << std::endl;
    std::cout << exp2 << std::endl;

    return 0;
}