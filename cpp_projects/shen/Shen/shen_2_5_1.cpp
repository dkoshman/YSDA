#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

void print_2_5_1(vector<int> v){
    for (int i : v)
        cout << i;
    cout << endl;
}

int shen_2_5_1(){
    cout << "Print all arrays of size k consisting of numbers 1..n in a chain "
                "of smallest possible changes" << endl;
    int n = 0, k = 0;
    cout << "Enter n, k: ";
#ifdef QT_DEBUG
    n = 4, k = 2;
#else
    cin >> n >> k;
#endif
    vector<int> v;
    vector<bool> d;
    for (int i = 0; i < n; ++i){
        v.push_back(1);
        d.push_back(true);
    }
//    Invariant: v[i] increases if d[i] is true, decreases otherwise.
//    On each cycle we change the rightmost possible element in the direction allowed by d, after which we switch
//    all d[i] to the right.
    while (true) {
        print_2_5_1(v);
        int t = n - 1;
        while (t >= 0 && (d[t] ? v[t] == k : v[t] == 1))
            --t;
        if (t < 0)
            break;
        v[t] += d[t] ? 1 : -1;
        for (int i = t + 1; i < n; ++i)
            d[i] = ~d[i];
    }

    return 0;
}



