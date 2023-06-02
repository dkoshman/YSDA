#include <iostream>
#include <numeric>
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
Rational operator*(const Rational& lhv, const Rational& rhv) {
    return Rational(lhv.Numerator() * rhv.Numerator(), lhv.Denominator() * rhv.Denominator());
}
Rational operator/(const Rational& lhv, const Rational& rhv) {
    if (rhv.Numerator() == 0) {
        throw std::domain_error("");
    }
    return Rational(lhv.Numerator() * rhv.Denominator(), lhv.Denominator() * rhv.Numerator());
}

ostream& operator<<(ostream& out, const Rational& rational) {
    out << rational.Numerator() << '/' << rational.Denominator();
    return out;
}

istream& operator>>(istream& in, Rational& rational) {
    int numerator, denominator;
    char c;
    in >> numerator;
    in.get(c);
    in >> denominator;
    if (not in || c != '/') {
        throw invalid_argument("");
    }
    rational = Rational(numerator, denominator);
    return in;
}
bool operator<(const Rational& lhv, const Rational& rhv) {
    return lhv.Numerator() * rhv.Denominator() < lhv.Denominator() * rhv.Numerator();
}
int main() {
    Rational lhv, rhv;
    char operation;
    try {
        cin >> lhv >> operation >> rhv;
        if (operation == '+') {
            cout << lhv + rhv;
        } else if (operation == '-') {
            cout << lhv - rhv;
        } else if (operation == '*') {
            cout << lhv * rhv;
        } else if (operation == '/') {
            cout << lhv / rhv;
        }
    } catch (invalid_argument& e) {
        cout << "Invalid argument\n";
    } catch (domain_error& e) {
        cout << "Division by zero\n";
    }

    return 0;
}