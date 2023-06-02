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

bool operator<(const Rational& lhv, const Rational& rhv) {
    return lhv.Numerator() * rhv.Denominator() < lhv.Denominator() * rhv.Numerator();
}

// Реализуйте для класса Rational оператор(ы), необходимые для использования его
// в качестве ключа map'а и элемента set'а

int main() {
    {
        const set<Rational> rs = {{1, 2}, {1, 25}, {3, 4}, {3, 4}, {1, 2}};
        if (rs.size() != 3) {
            cout << "Wrong amount of items in the set" << endl;
            return 1;
        }

        vector<Rational> v;
        for (auto x : rs) {
            v.push_back(x);
        }
        if (v != vector<Rational>{{1, 25}, {1, 2}, {3, 4}}) {
            cout << "Rationals comparison works incorrectly" << endl;
            return 2;
        }
    }

    {
        map<Rational, int> count;
        ++count[{1, 2}];
        ++count[{1, 2}];

        ++count[{2, 3}];

        if (count.size() != 2) {
            cout << "Wrong amount of items in the map" << endl;
            return 3;
        }
    }

    cout << "OK" << endl;
    return 0;
}
