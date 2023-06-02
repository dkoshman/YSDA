#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

class Numeral{
public:
    int n;
    vector<int> v;

    Numeral(int n, int x): n{n} {
        if (x == 0)
            v.push_back(0);
        while (x > 0) {
            v.push_back(x % n);
            x /= n;
        }
    }

    Numeral operator+(const Numeral& num) const {
        Numeral res(n, 0);
        int i = 0;
        res.v[i] = (v[i] + num.v[i]) % n;
        bool inc = (v[i] + num.v[i]) >= n;
        while (++i < max(v.size(), num.v.size())){
            if (i > v.size() - 1){
                res.v.push_back((num.v[i] + inc) % n);
                inc = (num.v[i] + inc) >= n;
                continue;
            }
            if (i > num.v.size() - 1){
                res.v.push_back((v[i] + inc) % n);
                inc = (v[i] + inc) >= n;
                continue;
            }
            res.v.push_back((v[i] + num.v[i] + inc) % n);
            inc = (v[i] + num.v[i] + inc) >= n;
        }
        if (inc)
            res.v.push_back(1);
        return res;
    }

    Numeral& operator++(){
        Numeral num(n, 1);
        *this = *this + num;
        return *this;
    }

    Numeral operator++(int){
        Numeral old = *this;
        operator++();
        return old;
    }

    bool operator<(const Numeral num) const {
        if (v.size() > num.v.size())
            return false;
        if (v.size() < num.v.size())
            return true;
        for (int i = v.size() - 1; i >= 0; --i)
            if (v[i] > num.v[i])
                return false;
            else
                if (v[i] < num.v[i])
                    return true;
        return false;
    }
};

ostream& operator<<(ostream& os, const Numeral& num){
    for (int i = num.v.size() - 1; i >= 0; --i)
        os << num.v[i];
    return os;
}

int shen_2_1_1(){
    cout << "Print all arrays of size k consisting of numbers 1..n" << endl;
    int n = 0, k = 0;
    cout << "Enter n, k: ";
#ifdef QT_DEBUG
    n = 2, k = 4;
#else
    cin >> n >> k;
#endif
    Numeral max(n, 0);
    Numeral inc(n + 1, 1);
    for (int i = 0; i < k - 1; ++i){
        max.v.push_back(0);
        inc.v.push_back(1);
    }
    max.v.push_back(1);
    Numeral num(n, 0);
    while (num < max)
        cout << (inc + num++) << endl;
    return 0;
}



