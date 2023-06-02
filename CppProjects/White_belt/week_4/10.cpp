#include <iostream>
#include <numeric>
using namespace std;

class Rational {
public:
    Rational() {
        // Реализуйте конструктор по умолчанию
    }

    Rational(int numerator, int denominator) {
        if (numerator == 0) {
            return;
        }
        int gcd = std::gcd(numerator, denominator);
        numerator /= gcd;
        denominator /= gcd;
        if (denominator < 0) {
            numerator_ = -numerator;
            denominator_ = -denominator;
        } else {
            numerator_ = numerator;
            denominator_ = denominator;
        }
        // Реализуйте конструктор
    }

    int Numerator() const {
        return numerator_;
        // Реализуйте этот метод
    }

    int Denominator() const {
        return denominator_;
        // Реализуйте этот метод
    }

private:
    int numerator_ = 0;
    int denominator_ = 1;
};

bool operator==(const Rational& lhv, const Rational& rhv) {
    return lhv.Numerator() == rhv.Numerator() && lhv.Denominator() == rhv.Denominator();
}

Rational operator-(const Rational& lhv, const Rational& rhv) {
    return Rational(lhv.Numerator() * rhv.Denominator() - lhv.Denominator() * rhv.Numerator(),
                    lhv.Denominator() * rhv.Denominator());
}
Rational operator+(const Rational& lhv, const Rational& rhv) {
    return Rational(lhv.Numerator() * rhv.Denominator() + lhv.Denominator() * rhv.Numerator(),
                    lhv.Denominator() * rhv.Denominator());
}
// Реализуйте для класса Rational операторы ==, + и -

int main() {
    {
        Rational r1(4, 6);
        Rational r2(2, 3);
        bool equal = r1 == r2;
        if (!equal) {
            cout << "4/6 != 2/3" << endl;
            return 1;
        }
    }

    {
        Rational a(2, 3);
        Rational b(4, 3);
        Rational c = a + b;
        bool equal = c == Rational(2, 1);
        if (!equal) {
            cout << "2/3 + 4/3 != 2" << endl;
            return 2;
        }
    }

    {
        Rational a(5, 7);
        Rational b(2, 9);
        Rational c = a - b;
        bool equal = c == Rational(31, 63);
        if (!equal) {
            cout << "5/7 - 2/9 != 31/63" << endl;
            return 3;
        }
    }

    cout << "OK" << endl;
    return 0;
}
