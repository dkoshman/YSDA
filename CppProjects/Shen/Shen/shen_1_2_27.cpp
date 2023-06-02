#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

int shen_1_2_27(){
    cout << "Perform a binary search of x in a random array of size n" << endl;
    int n = 0, x = 0;
    cout << "Enter n, x: ";
    cin >> n >> x;
    vector<int> v;
    mt19937 mt(time(nullptr));
    for (int i = 0; i < n; ++i)
        v.push_back(mt() % n);
    sort(v.begin(), v.end());
    for (int i : v)
        cout << i << ' ';
    cout << endl;
    int a = 0, b = n, c = 0;
//    Invariant: x is between v[a] and v[b - 1]
    while (a != b - 1) {
        c = (a + b) / 2;
        if (x < v[c])
            b = c;
        else
            a = c;
    }
    cout << a << ' ' << v[a] << endl;
    return 0;
}
