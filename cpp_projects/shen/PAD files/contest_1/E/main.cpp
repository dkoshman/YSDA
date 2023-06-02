#include <iostream>

using namespace std;

int main()
{
    int n = 0;
    int k = 0;
    int m = 0;
    int k_m = 0;
    int n_0 = 0;

    cin >> n >> k >> m;
    if ((k > n) || (m > k)){
        cout << 0;
    } else {
        k_m = (k / m) * m;
        n_0 = n;
        while (n >= k){
            n -= (n / k) * k_m;
        }
        cout << (n_0 - n) / m;
    }
    return 0;
}
