#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

void print_2_6_1(vector<int> v) {
    for (int i : v)
        cout << (i == -1 ? 0 : i);
    cout << endl;
}

int factorial_2_6_1(int n) {
    int r = 1;
    for (int i = 1; i <= n; ++ i)
        r *= i;
    return r;
}

int shen_2_6_1(){
    cout << "Print all sequences corresponding to Catalan numbers: modeled by a 2n queue in cinema that doesn't stall or "
            "by sequences of 2n of 1 and -1 such that cumulative sum is non negative" << endl;
    int n = 0;
    cout << "Enter n: ";
#ifdef QT_DEBUG
    n = 4;
#else
    cin >> n;
#endif
    vector<int> v;
    for (int i = 0; i < 2 * n; ++i)
        v.push_back(i % 2 ? -1 : 1);
    int catalan = 0;
//    Invariant: printed catalan sequences of ±1 go in lexicographical order, all possible sequences that are lexicographically
//    junior to the last one are printed. So next possible sequence is generated from last by seeking rightmost possible change
//    from -1 to 1, ie -1 that has 1 after it, subsequence after that change is set to least possible.
    while (true) {
        print_2_6_1(v);
        ++catalan;
        int k = 2 * n - 1, sum = 0;
        bool followed_by_1 = false;
        while (k > 0 && not (followed_by_1 && v[k] == -1)) {
            sum += v[k];
            followed_by_1 = v[k] == 1 ? true : followed_by_1;
            --k;
        }
        if (k == 0)
            break;
        v[k] = 1;
        sum = -sum + 2;
        for (int i = k + 1; i < 2 * n; ++i) {
            if (sum > 0) {
                v[i] = -1;
                --sum;
            } else {
                v[i] = 1;
                ++sum;
            }
        }
    }
    assert(catalan == (factorial_2_6_1(2 * n) / factorial_2_6_1(n) / factorial_2_6_1(n) / (n + 1)));
    cout << n << "th catalan number is " << catalan << endl;
    return 0;
}



