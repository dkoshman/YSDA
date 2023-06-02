#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

void print_2_2_1(vector<int> const &v){
    for (int i : v)
        cout << i;
    cout << endl;
}

int shen_2_2_1(){
    cout << "Print all permutations of size n" << endl;
    int n = 0;
    cout << "Enter n: ";
#ifdef QT_DEBUG
    n = 4;
#else
    cin >> n;
#endif
    vector<int> v;
    for (int i = 1; i <= n; ++i)
        v.push_back(i);

    int count = 0;
//    Invariant: v[k] is the leftmost element after which array is decreasing
    while (true){
        print_2_2_1(v);
        ++count;
        int k = n - 2;
        while (k >= 0 && v[k] > v[k + 1])
            --k;
        if (k < 0)
            break;
        int t = k + 1;
        while (t < n && v[k] < v[t])
            ++t;
        iter_swap(v.begin() + k, v.begin() + t - 1);
        sort(v.begin() + k + 1, v.end());
    }
    int count_check = 1;
    for (int i = 1; i <= n; ++i)
        count_check *= i;
    assert(count == count_check);
    return 0;
}



