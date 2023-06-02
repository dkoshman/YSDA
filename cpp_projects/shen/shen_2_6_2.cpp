#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

void close_brackets(vector<int> &lvl, int &cur) {
    while (lvl[cur] == 2) {
        cout << ')';
        lvl[cur] = 0;
        lvl[--cur] += 1;
    }
}

void print_2_6_2(vector<int> v){
    char c = 'a';
    int cur = 0;
    vector<int> lvl;
    for (int _ : v)
        lvl.push_back(0);
//    Invariant: current amount of open brackets is $cur$, the number of complete expressions on level i is lvl[i]
    for (int i : v){
        if (i == 1){
            cout << '(';
            ++cur;
        } else {
            cout << c++;
            ++lvl[cur];
            close_brackets(lvl, cur);
        }
    }
    cout << c;
    ++lvl[cur];
    close_brackets(lvl, cur);
    cout << endl;
}

int shen_2_6_2(){
    cout << "Print all ways to sequence multiplication between n numbers standing in fixed order. "
            "This task is isomorphic to shen_2_6_1, the only difference is in print function" << endl;
    int n = 0;
    cout << "Enter n: ";
#ifdef QT_DEBUG
    n = 4;
#else
    cin >> n;
#endif
//    Because ways to sequence multiplication of n elements is isomorphic to n - 1 st cardinal number sequence, I decrement it.
    --n;
    vector<int> v;
    for (int i = 0; i < 2 * n; ++i)
        v.push_back(i % 2 ? -1 : 1);

    while (true) {
        print_2_6_2(v);
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
    return 0;
}



