#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

void print_2_4_1(vector<int> v) {
    bool first = true;
    for (int i : v){
        if (i == 0)
            break;
        if (first){
            cout << i;
            first = false;
        }
        else
            cout << '+' << i;
    }
    cout << endl;
}

int shen_2_4_1(){
    cout << "Print all decompositions of n into positive terms" << endl;
    int n = 0;
    cout << "Enter n: ";
#ifdef QT_DEBUG
    n = 8;
#else
    cin >> n;
#endif
    vector<int> v;
    for (int i = 0; i < n; ++i)
        v.push_back(1);

    int s = n - 1;
//    Invariant: v is non increasing, sum of v[i] is n, order of generating decompositions is lexicographical,
//    v[s] is the rightmost non zero element.
    while (true) {
        print_2_4_1(v);
        if (v[0] == n)
            break;
        int t = s - 1;
        int sum = v[s];
        while (t > 0 && v[t - 1] == v[t]){
            sum += v[t];
            --t;
        }
        s = t + sum - 1;
        ++v[t];
        for (int i = t + 1; i < n; ++i)
            if (i < t + sum)
                v[i] = 1;
            else
                v[i] = 0;
    }
    return 0;
}



