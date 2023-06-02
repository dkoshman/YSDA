#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

int shen_1_2_13(){
    cout << "Compute value of polynomial of degree n and its derivative at point x" << endl;
    int n = 0, x = 0;
    cout << "Enter n, x: ";
#ifdef QT_DEBUG
    n = 4;
#else
    cin >> n >> x;
#endif
    mt19937 mt(time(nullptr));
    vector<int> v;
    for (int i = 0; i <= n; ++i)
        v.push_back(n - mt() % n * 2);
    int val = 0, der = 0;
    for (int i = 0; i <= n; ++i) {
        der = x * der + val;
        val = val * x + v[i];
    }
    for (int i = 0; i <= n; ++i) {
        if (i != 0)
            cout << " + ";
        cout << v[i] << "x^" << n - i;
    }
    cout << endl << "Its value at " << x << " is " << val << ", its derivative is " << der << endl;
    return 0;
}



