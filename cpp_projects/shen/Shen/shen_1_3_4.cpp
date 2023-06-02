#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

int shen_1_3_4(){
    cout << "Find longest increasing subarray in array of size n." << endl;
    int n = 0;
    cout << "Enter n: ";
#ifdef QT_DEBUG
    n = 10;
#else
    cin >> n;
#endif
    vector<int> v;
    mt19937 mt(time(nullptr));
    for (int i = 0; i < n; ++i)
        v.push_back(mt() % n);

    for (int i : v)
        cout << i << ' ';
    cout << endl;

    vector<int> u;
//    Invariant: size of u is the length of longest increasing subarray, u[i] is the smallest possible last
//    element of increasing subarray of size i + 1.
    for (int i : v){
        auto pos = lower_bound(u.begin(), u.end(), i);
        if (pos == u.end())
            u.push_back(i);
        else
            *pos = i;
    }
    cout << "Largest size of increasing subarray is " << u.size() << endl;
#ifdef QT_DEBUG
    for (int i : u)
        cout << i << ' ';
    cout << endl;
#endif

    u.clear();
    for (int i : v){
//        Only works for integers
        auto pos = lower_bound(u.begin(), u.end(), i + 1);
        if (pos == u.end())
            u.push_back(i);
        else
            *pos = i;
//        Works for doubles
//        auto pos = lower_bound(u.begin(), u.end(), i);
//        while (pos != u.end() && *pos == i)
//            ++pos;
//        if (pos == u.end())
//            u.push_back(i);
//        else
//            *pos = i;
//#ifdef QT_DEBUG
    for (int i : u)
        cout << i << ' ';
    cout << endl;
//#endif
    }
    cout << "Largest size of nondecreasing subarray is " << u.size() << endl;
    return 0;
}



