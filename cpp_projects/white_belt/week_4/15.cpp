#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <numeric>
#include <sstream>
using namespace std;

class Rational {
public:
    Rational() {
        // Реализуйте конструктор по умолчанию
    }

    Rational(int numerator, int denominator) {
        if (denominator == 0) {
            throw std::invalid_argument("");
        }
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

    Rational& operator=(const Rational& other) {
        if (this == &other) {
            return *this;
        }
        numerator_ = other.Numerator();
        denominator_ = other.Denominator();
        return *this;
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

Rational operator/(const Rational& lhv, const Rational& rhv) {
    if (rhv.Numerator() == 0) {
        throw std::domain_error("");
    }
    return Rational(lhv.Numerator() * rhv.Denominator(), lhv.Denominator() * rhv.Numerator());
}
// Вставьте сюда реализацию operator / для класса Rational
//
//int main() {
//    try {
//        Rational r(1, 0);
//        cout << "Doesn't throw in case of zero denominator" << endl;
//        return 1;
//    } catch (invalid_argument&) {
//    }
//
//    try {
//        auto x = Rational(1, 2) / Rational(0, 1);
//        cout << "Doesn't throw in case of division by zero" << endl;
//        return 2;
//    } catch (domain_error&) {
//    }
//
//    cout << "OK" << endl;
//    return 0;
//}
