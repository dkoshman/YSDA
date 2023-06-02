#include <iostream>

using namespace std;
typedef unsigned long long shlong;

int main()
{
    shlong a = 0;
    shlong b = 0;
    shlong res = 1;
    shlong p = 1000000007;

    cin >> a >> b;
    while (b > 0){
        if (b % 2){
            res = (res * a) % p;
            --b;
        } else {
            a = (a * a) % p;
            b /= 2;
        }
    }
    cout << res;
    return 0;
}
