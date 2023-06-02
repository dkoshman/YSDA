#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

void print_1_2_32(vector<int> &v){
    for (int i : v)
        cout << i << ',';
    cout << endl;
}

int shen_1_2_32(){
    cout << "Partially sort array in three parts: less than x, equal x and more than x" << endl;
    int n = 0;
    cout << "Enter size of array n: ";
#ifdef QT_DEBUG
    n = 4;
#else
    cin >> n;
#endif
    vector<int> v;
    mt19937 mt(time(nullptr));
    for (int i = 0; i < n; ++i)
        v.push_back(mt() % n);
    int less_id = 0, eq_id = 0, unknown_id = n;
//    Invariant v: [...<n/2...less_id...==n/2...eq_id...???...unknown_id...>n/2...]
    while (unknown_id != eq_id) {
        if (v[eq_id] < n / 2) {
            iter_swap(v.begin() + less_id, v.begin() + eq_id);
            ++less_id;
            ++eq_id;
        } else if (v[eq_id] == n / 2) {
            ++eq_id;
        } else {
            --unknown_id;
            iter_swap(v.begin() + eq_id, v.begin() + unknown_id);
        }
    }
    print_1_2_32(v);

    return 0;
}



