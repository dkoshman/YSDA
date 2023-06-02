//https://contest.yandex.ru/contest/27631/problems/E/
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using shlong = unsigned long long;
using namespace std;

class Prime{
public:
    int n, p;
    vector<bool> v;
    Prime(int n) : n{n}{
        v.resize(n + 1);
        v[0] = v[1] = true;
        p = 2;
    }
    int next(){
        while (v[p] == true && p <= n)
            ++p;
        if (p > n)
            return -1;
        int i = p;
        while (i <= n) {
            v[i] = true;
            i += p;
        }
        return p;
    }
};

shlong gen_1(int n){
    shlong res = 1;
    for (int i = 1; i < n; ++i)
        res = res * 10 + 1;
    return res;
}
shlong gen_pow(int n){
    shlong res = 1;
    for (int i = 1; i < n; ++i)
        res *= 10;
    return res;
}
int main()
{
    string s;
    getline(cin, s);
    stringstream ss(s);
    char c;
    int sum = 0;
    int n = 0;
    while (ss.good()) {
        ss >> c;
        if (not ss.good())
                break;
        sum += c - '0';
        ++n;
    }
    Prime P(sum + 10);
    int x = P.next();
    while (sum % x != 0) {
        x = P.next();
    }
    Prime P2(sum + 10);
    int y = P2.next();
    int y_s = 15;
    shlong uno = gen_1(y_s);
    shlong p10 = gen_pow(y_s + 1);
//    cout << uno <<"uno" << endl;
//    cout << p10 <<"p10" << endl;
    while (y < x) {
        int k = n;
        shlong z = 0;
        shlong p10_y = p10 % y;
        shlong uno_y = uno % y;
        while (k > 0) {
            if (k >= y_s){
                z = (z * p10_y + uno_y) % y;
                k -= y_s;
            } else {
                z = (z * gen_pow(k + 1) + gen_1(k)) % y;
                k = 0;
            }
        }
        if (z == 0)
            break;
        y = P2.next();
    }
//    cout << sum << ' ' << n << ' ' << x << ' ' << y << endl;
    cout << min(x, y) << endl;
    return 0;
}
