#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

void print_2_3_1(vector<int> const &v){
    for (int i : v)
        cout << i;
    cout << endl;
}

int factorial_2_3_1(int n){
    int res = 1;
    for (int i = 2; i <= n; ++i)
        res *= i;
    return res;
}

int shen_2_3_1(){
    cout << "Print all k element subsets of 1..n" << endl;
    int n = 0, k = 0;
    cout << "Enter n, k: ";
#ifdef QT_DEBUG
    n = 4, k = 2;
#else
    cin >> n >> k;
#endif
    vector<int> v;
    for (int i = 0; i < n; ++i)
        v.push_back(i < n - k ? 0 : 1);

    int count = 0;
//    Invariant: v[t] is the rightmost element so that v[t] = 0, v[t + 1] = 1
    while (true) {
        int t = n - 2;
        print_2_3_1(v);
        ++count;
        while (t >= 0 && v[t] >= v[t + 1])
            --t;
        if (t < 0)
            break;
        iter_swap(v.begin() + t, v.begin() + t + 1);
        sort(v.begin() + t + 1, v.end());
    }
    assert(count == factorial_2_3_1(n) / factorial_2_3_1(k) / factorial_2_3_1(n - k));

    return 0;
}
