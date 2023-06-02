#include <iostream>

using namespace std;

class Rational{
public:
    int num, den;
    Rational(int m = 0, int n = 1): num(m), den(n){
        pair<int, int> p = this->normalize();
        num = p.first;
        den = p.second;
    };
    pair<int, int> normalize() const {
        int n = num > 0 ? num : -num;
        int m = den > 0 ? den : -den;
        int t = 0;

        while (n % m != 0) {
            t = m;
            m = n % m;
            n = t;
        }
        n = ((den > 0) ? (num / m) : (-num / m));
        m = ((den > 0) ? (den / m) : (-den / m));
        return make_pair(n, m);
    }
    int numerator() const {
        return this->normalize().first;
    };
    int denominator() const {
        return this->normalize().second;
    };
    Rational operator+(int x) const {
        return Rational(num + x*den, den);
    }
    Rational operator+(const Rational& x) const {
        return Rational(num*x.den + den*x.num, den*x.den);
    }
    Rational operator+() const {
        return *this;
    }
    Rational operator-(int x) const {
        return Rational(num - x*den, den);
    }
    Rational operator-(const Rational& x) const {
        return Rational(num*x.den - den*x.num, den*x.den);
    }
    Rational operator-() const {
        return Rational(-num, den);
    }
    Rational operator*(int x) const {
        return Rational(num * x, den);
    }
    Rational operator*(const Rational& x) const {
        return Rational(num * x.num, den * x.den);
    }
    Rational operator/(int x) const {
        return Rational(num, den * x);
    }
    Rational operator/(const Rational & x) const {
        return Rational(num * x.den, den * x.num);
    }
    void operator=(const Rational & x){
        int n = x.numerator();
        int m = x.denominator();
        num = n;
        den = m;
    }
    bool operator==(const Rational& x) const {
        return (this->numerator() == x.numerator() && this->denominator() == x.denominator());
    }
    bool operator!=(const Rational& x) const {
        return not(*this == x);
    }
    Rational& operator+=(const Rational& x){
        *this = *this + x;
        return *this;
    }
    Rational& operator-=(const Rational& x){
        *this = *this - x;
        return *this;
    }
    Rational& operator*=(const Rational& x){
        *this = *this * x;
        return *this;
    }
    Rational& operator/=(const Rational& x){
        *this = *this / x;
        return *this;
    }
    Rational& operator++(){
        *this += Rational(1);
        return *this;
    }
    Rational operator++(int){
        Rational q = *this;
        ++(*this);
        return q;
    }
    Rational& operator--(){
        *this -= Rational(1);
        return *this;
    }
    Rational operator--(int){
        Rational q = *this;
        --(*this);
        return q;
    }
};

Rational operator+(int x, const Rational& y){
    return Rational(x) + y;
};

Rational operator-(int x, const Rational& y){
    return Rational(x) - y;
};

Rational operator*(int x, const Rational& y){
    return Rational(x) * y;
};

Rational operator/(int x, const Rational& y){
    return Rational(x) / y;
};

//void print(Rational q){
//    cout << q.num << '/' << q.den << '(' << q.numerator() << '/' << q.denominator() << ')' << endl;
//}

//void check(bool b){
//    cout << (b ? "OK" : "ERROR") << endl;
//}

//int main(){
//    Rational q(-3, 5);
//    Rational x(12, -21);
//    Rational c(1);
//    check(q == (q + 1 - c));
//    c = x;
//    x /= q;
//    check(x * q == c);
//    c = x++;
//    --x;
//    check(c == x);
//    check(++x == x);
//    check(--x == x);
//    check(3 * x++ == --x * 3);
//    check(12 / x == Rational(12) /x);
//    check(x / 5 == Rational(1, 5) * x);
//    return 0;
//}
