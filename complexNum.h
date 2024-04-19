#ifndef COMPLEX_NUM_H
#define COMPLEX_NUM_H

#include <cmath>
#include <iostream>

/*
Complex number class
Implements basic arithmetic operations, as well as absolute value, exponential and conjugate functions
*/
template <typename T>
struct Complex {
    T real;
    T imag;

    // Constructor, defaults to 0 + 0i
    Complex(T r = T(), T i = T()) : real(r), imag(i) {}

    Complex operator+(const Complex& other) const {
        /**
         * Addition of two complex numbers
         * 
         * defined as:
         * (a + bi) + (c + di) = (a + c) + (b + d)i
         *
         * @param other: Complex number to add
         * @return: Sum of the two complex numbers
        */
        return Complex(real + other.real, imag + other.imag);
    }

    friend Complex<T> operator+(T real, const Complex& c) {
        /**
         * Addition of a real number and a complex number
         * 
         * defined as:
         * a + (b + ci) = (a + b) + ci
         *
         * @param real: Real number to add
         * @param c: Complex number to add
         * @return: Sum of the real and complex numbers
        */
        return Complex(real + c.real, c.imag);
    }

    friend Complex<T> operator+(const Complex& c, T real) {
        /**
         * Addition of a complex number and a real number
         * 
         * defined as:
         * (a + bi) + c = (a + c) + bi
         *
         * @param c: Complex number to add
         * @param real: Real number to add
         * @return: Sum of the complex and real numbers
        */
        return Complex(c.real + real, c.imag);
    }

    Complex operator-(const Complex& other) const {
        /**
         * Subtraction of two complex numbers
         * 
         * defined as:
         * (a + bi) - (c + di) = (a - c) + (b - d)i
         *
         * @param other: Complex number to subtract
         * @return: Difference of the two complex numbers
        */
        return Complex(real - other.real, imag - other.imag);
    }

    friend Complex<T> operator-(T real, const Complex& c) {
        /**
         * Subtraction of a real number and a complex number
         * 
         * defined as:
         * a - (b + ci) = (a - b) - ci
         *
         * @param real: Real number to subtract
         * @param c: Complex number to subtract
         * @return: Difference of the real and complex numbers
        */
        return Complex(real - c.real, -c.imag);
    }

    friend Complex<T> operator-(const Complex& c, T real) {
        /**
         * Subtraction of a complex number and a real number
         * 
         * defined as:
         * (a + bi) - c = (a - c) + bi
         *
         * @param c: Complex number to subtract
         * @param real: Real number to subtract
         * @return: Difference of the complex and real numbers
        */
        return Complex(c.real - real, c.imag);
    }

    Complex operator*(const Complex& other) const {
        /**
         * Multiplication of two complex numbers
         * 
         * defined as (distributive property):
         * (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
         *
         * @param other: Complex number to multiply
         * @return: Product of the two complex numbers
        */
        return Complex(real * other.real - imag * other.imag,
                       real * other.imag + imag * other.real);
    }

    friend Complex<T> operator*(T real, const Complex& c) {
        /**
         * Multiplication of a real number and a complex number
         * 
         * defined as:
         * a * (b + ci) = ab + aci
         *
         * @param real: Real number to multiply
         * @param c: Complex number to multiply
         * @return: Product of the real and complex numbers
        */
        return Complex(real * c.real, real * c.imag);
    }

    friend Complex<T> operator*(const Complex& c, T real) {
        /**
         * Multiplication of a complex number and a real number
         * 
         * defined as:
         * (a + bi) * c = ac + bci
         *
         * @param c: Complex number to multiply
         * @param real: Real number to multiply
         * @return: Product of the complex and real numbers
        */
        return Complex(c.real * real, c.imag * real);
    }

    Complex operator/(const Complex& other) const {
        /**
         * Division of two complex numbers
         * 
         * defined as:
         * (a + bi) / (c + di) = (a + bi) * (c - di) / (c^2 + d^2)
         *
         * @param other: Complex number to divide by
         * @return: Quotient of the two complex numbers
        */
        T denominator = other.real * other.real + other.imag * other.imag;
        if (denominator == 0) {
            throw std::invalid_argument("Division by zero");
        }
        return Complex(
            (real * other.real + imag * other.imag) / denominator,
            (imag * other.real - real * other.imag) / denominator
        );
    }

    friend Complex<T> operator/(T real, const Complex& c) {
        /**
         * Division of a real number by a complex number
         * 
         * defined as:
         * a / (b + ci) = a * (b - ci) / (b^2 + c^2)
         *
         * @param real: Real number to divide
         * @param c: Complex number to divide by
         * @return: Quotient of the real and complex numbers
        */
        T denominator = c.real * c.real + c.imag * c.imag;
        if (denominator == 0) {
            throw std::invalid_argument("Division by zero");
        }
        return Complex(real * c.real / denominator, -real * c.imag / denominator);
    }

    friend Complex<T> operator/(const Complex& c, T real) {
        /**
         * Division of a complex number by a real number
         * 
         * defined as:
         * (a + bi) / c = a / c + bi / c
         *
         * @param c: Complex number to divide
         * @param real: Real number to divide by
         * @return: Quotient of the complex and real numbers
        */
        if (real == 0) {
            throw std::invalid_argument("Division by zero");
        }
        return Complex(c.real / real, c.imag / real);
    }

    T abs() const {
        /**
         * Absolute value (magnitude) of a complex number
         * 
         * defined as:
         * |a + bi| = sqrt(a^2 + b^2)
         *
         * @return: Absolute value of the complex number
        */
        return std::sqrt(real * real + imag * imag);
    }

    Complex exp() const {
        /**
         * Exponential of a complex number
         * 
         * defined as (Euler's formula):
         * exp(a + bi) = exp(a) * (cos(b) + sin(b)i)
         *
         * @return: Exponential of the complex number
        */
        T expReal = std::exp(real);
        return Complex(expReal * std::cos(imag), expReal * std::sin(imag));
    }

    Complex conj() const {
        /**
         * Conjugate of a complex number
         * 
         * defined as:
         * conj(a + bi) = a - bi
         *
         * @return: Conjugate of the complex number
        */
        return Complex(real, -imag);
    }

    friend std::ostream& operator<<(std::ostream& os, const Complex& c) {
        /**
         * Overload the << operator to print the complex number
         * 
         * @param os: Output stream
         * @param c: Complex number to print
         * @return: Output stream with the complex number printed
        */
        os << "(" << c.real << " + " << c.imag << "i)";
        return os;
    }
};

#endif // COMPLEX_NUM_H