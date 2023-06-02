/*В этой задаче вам нужно вычислить значение функции Эйлера от некоторого биномиального коэффициента
 *  («выбрать k элементов из n»)*/

#include <iostream>
#include <vector>

using namespace std;
using shlong = unsigned long long;
shlong MOD = 1000000007;

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

int main()
{
    int k = 0, n = 0, p = 0, p_count = 0;
    shlong p_count_mod = 0, res = 1, p_pow = 0;
    cin >> k >> n;
    Prime P(n);
    p = P.next();
    while (p != -1){
        p_pow = p;
        p_count = 0;
        while (p_pow <= (shlong)n){
            p_count += n / p_pow - k / p_pow - (n - k) / p_pow;
            p_pow *= p;
        }
        if (p_count == 0){
            p = P.next();
            continue;
        }
        p_count_mod = 1;
        for (int i = 0; i < p_count - 1; ++i)
            p_count_mod = (p_count_mod * p) % MOD;
        p_count_mod = (p_count_mod * (p - 1)) % MOD;
        res = (res * p_count_mod) % MOD;
        p = P.next();
    }
    cout << res << endl;
    return 0;
}
