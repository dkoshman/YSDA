#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

void print_(vector<int> v){
    for (int i : v)
        cout << i;
    cout << endl;
}

int shen_1_1_29(){
    cout << "Count number of natural solutions to x^2 + y^2 < n" << endl;
    int n = 0;
    cout << "Enter n: ";
#ifdef QT_DEBUG
    n = 4;
#else
    cin >> n;
#endif
    int k = 0, l = 0, s = 0;
    while (l * l < n) {
        ++l;
    }
//    Invariant: l is the smallest natural number so that k, l is not a solution to x^2 + y^2 < n,
//    s is the number of solutions for x < k
    while (l != 0) {
        s += l;
        cout << s << ' ';
        ++k;
        while ((l != 0) and (k * k + (l - 1) * (l - 1) > n)) {
            --l;
        }
    }
    cout << "Number of solutions is " << s << endl;

    return 0;
}



