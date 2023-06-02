#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

int shen_1_2_30(){
    cout << "Determine parity of permutation of size n, find its inverse" << endl;
    int n = 0;
    cout << "Enter n: ";
#ifdef QT_DEBUG
    n = 8;
#else
    cin >> n;
#endif
    vector<int> v;
    for (int i = 0; i < n; ++i)
        v.push_back(i + 1);
    mt19937 mt(time(nullptr));
    for (int i = 0; i < n; ++i)
        iter_swap(v.begin() + i, v.begin() + mt() % n);
    for (int i: v)
        cout << i << ' ';
    cout << endl;

    vector<int> v_i(v.size());
    bool parity = false;
    int i = 0, start = 0, cycle = 1;
//    Invariant: v[i] is in a cycle starting at $start$ of at least size $cycle$, we passed over elements
//    in v whose value in v_i is equal to 1.
    while (i < n) {
        v_i[i] = 1;
        if (v[i] != start + 1){
            ++cycle;
            i = v[i] - 1;
        }
        else {
            parity = parity xor ((cycle + 1) % 2);
            cycle = 1;
            while (start < n && v_i[start] == 1)
                ++start;
            i = start;
        }
    }
    cout << "Permutation is " << (parity ? "odd" : "even") << ". It's reverse is:" << endl;

    for (size_t j{0}; j < v.size(); ++j)
        v_i[v[j] - 1] = j + 1;
    for (int i : v_i)
        cout << i << ' ';
    cout << endl;
    return 0;
}
