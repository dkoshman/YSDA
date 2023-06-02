#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

void print_2_5_2(vector<int> v){
    for (int i : v)
        cout << i;
    cout << endl;
}

int shen_2_5_2(){
    cout << "Print all permutations of 1..n in a chain "
        "of consecutive transpositions" << endl;
    int n = 0;
    cout << "Enter n: ";
#ifdef QT_DEBUG
    n = 4;
#else
    cin >> n;
#endif
    vector<int> v, x, y;
    vector<bool> d;
    for (int i = 0; i < n; ++i) {
        v.push_back(i + 1);
        y.push_back(i);
        d.push_back(false);
        x.push_back(i);
    }

    int count = 0;
//    Invariant: y[i] is the number of elements less than i and standing to the left in permutation v.
//    I iterate through all possible d such that y[i] <= i via single increments and decrements, similarly to shen_2_5_1.
//    Each possible y is isomorphic to permutation v, and if all y[j] to the right of y[i] are max possible or min possible,
//    then there's always a possible transposition corresponding to increment or decrement of y[i]. x[i] is the position of
//    i + 1 in v: v[x[i]] = i + 1
    while (true) {
        print_2_5_2(v);
        ++count;
        int t = n - 1;
        while (t >= 0 && (d[t] ? y[t] == t : y[t] == 0))
            --t;
        if (t < 0)
            break;
        int dif = d[t] ? 1 : -1;
        y[t] += dif;
        iter_swap(v.begin() + x[t], v.begin() + x[t] + dif);
        x[v[x[t]] - 1] -= dif;
        x[t] += dif;
        for (int i = t + 1; i < n; ++i)
            d[i] = ~d[i];
    }
    int count_check = 1;
//    Check that the number of permutations printed is equal to n!
    for (int i = 1; i <= n; ++i)
        count_check *= i;
    assert(count == count_check);
    return 0;
}



