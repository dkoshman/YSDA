#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

void print_1_2_22(vector<int> &v){
    for (int i : v)
        cout << i << ", ";
    cout << endl;
}

int shen_1_2_22(){
    cout << "Given n and two nondecreasing arrays x and y, find closest sum x[i] + y[j] to n" << endl;
    int s = 0;
    cout << "Enter upper bound for random arrays size: ";
#ifdef QT_DEBUG
    s = 4;
#else
    cin >> s;
#endif
    mt19937 mt(time(nullptr));
    int x_size = (mt() % s) + 1, y_size = (mt() % s) + 1, n = mt() % (s * s / 2);
    vector<int> x, y;
    for (int i = 0; i < x_size; ++i)
        x.push_back(mt() % s);
    for (int i = 1; i < x_size; ++i)
        x[i] += x[i - 1];
    for (int i = 0; i < y_size; ++i)
        y.push_back(mt() % s);
    for (int i = 1; i < y_size; ++i)
        y[i] += y[i - 1];
    print_1_2_22(x);
    print_1_2_22(y);
    reverse(y.begin(), y.end());
    for (int &i : y)
        i = n - i;
    int x_id = 0, y_id = 0, x_id_res = 0, y_id_res = 0;
//    I find the closest sum by merging two nondecreasing arrays: x and reverse(n - y)
//    and in the process find the distance between these arrays
    while ((x_id != x_size - 1) or (y_id != y_size - 1)) {
        if (x_id == x_size - 1) {
            ++y_id;
        }
        else if (y_id == y_size - 1) {
            ++x_id;
        } else {
            if (x[x_id + 1] < y[y_id + 1])
                ++x_id;
            else
                ++y_id;
        }
        if (abs(x[x_id] - y[y_id]) < abs(x[x_id_res] - y[y_id_res])) {
            x_id_res = x_id;
            y_id_res = y_id;
        }
    }
    cout << x[x_id_res] << " + " << (n - y[y_id_res]) << " ~ " << n << endl;
    return 0;
}



