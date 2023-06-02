#include <iostream>
#include <math.h>

using namespace std;

class Complex{
    double real, imag;
public:
    Complex(double x){real = x; imag = 0;};
    Complex(double x, double w){real = x; imag = w;};
    double Re() const {return this->real;};
    double Im() const {return this->imag;};
    Complex operator+(double x){
        Complex C(real + x, imag);
        return C;
    }
    Complex operator+(Complex x){
        Complex C(real + x.Re(), imag + x.Im());
        return C;
    }
    Complex operator+(){
        return *this;
    }
    Complex operator-(double x){
        Complex C(real - x, imag);
        return C;
    }
    Complex operator-(Complex x){
        Complex C(real - x.Re(), imag - x.Im());
        return C;
    }
    Complex operator-(){
        Complex C(-real, -imag);
        return C;
    }
    Complex operator*(double x){
        Complex C(real * x, imag * x);
        return C;
    }
    Complex operator*(Complex x){
        Complex C(real * x.Re() - imag * x.Im(), real * x.Im() + imag * x.Re());
        return C;
    }
    Complex operator/(double x){
        Complex C(real / x, imag / x);
        return C;
    }
    Complex operator/(Complex x){
        Complex C((real*x.Re() + imag*x.Im()) / (x.Im()*x.Im() + x.Re()*x.Re()),
                  (imag*x.Re() - real*x.Im()) / (x.Im()*x.Im() + x.Re()*x.Re()));
        return C;
    }
    void operator=(Complex x){
        this->real = x.Re();
        this->imag = x.Im();
    }
    bool operator==(Complex x){
        return (this->imag == x.Im() && this->real == x.Re());
    }
    bool operator!=(Complex x){
        return not(*this == x);
    }
};

Complex operator+(double x, Complex y){
    Complex c(x);
    return c + y;
};

Complex operator-(double x, Complex y){
    Complex c(x);
    return c - y;
};

Complex operator*(double x, Complex y){
    Complex c(x);
    return c * y;
};

Complex operator/(double x, Complex y){
    Complex c(x);
    return c / y;
};

double abs (Complex c) {
    return sqrt(c.Re()*c.Re() + c.Im()*c.Im());
};
