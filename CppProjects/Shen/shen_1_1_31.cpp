#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

int shen_1_1_31(){
    cout << "Compute the period of 1/n fraction" << endl;
    int n = 0;
    cout << "Enter n: ";
#ifdef QT_DEBUG
    n = 4;
#else
    cin >> n;
#endif
    int k = 0, r = 1;
//    Length of pre-period is no more than n, so I skip it.
    while (k < n) {
        r = r * 10 % n;
        ++k;
    }
    int c = r;
    r = r * 10 % n;
    k = 1;
    while (c != r) {
        r = r * 10 % n;
        ++k;
    }
    cout << "Period of 1/" << n << " is " << k << endl;

    return 0;
}



