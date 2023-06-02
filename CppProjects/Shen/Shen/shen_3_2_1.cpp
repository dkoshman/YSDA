#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>


using namespace std;

class Tree_traversal{
public:
    vector<int> &v;
    vector<bool> b;
    int s, k;

    Tree_traversal(vector<int> &v, int s): v{v}, s{s} {}

    void begin_work(){
        k = -1;
        for (int _ : v)
            b.push_back(false);
    }
    int sum(){
        int sum = 0;
        for (int i = 0; i <= k; ++i)
            if (b[i])
                sum += v[i];
        return sum;
    }
    bool is_down(){
        return k >= 0;
    }
    bool is_right(){
        return b[k] == false;
    }
    bool is_up(){
        return k < (int)b.size() - 1 && sum() <= s;
    }
    void up(){
        ++k;
        b[k] = false;
    }
    void right(){
        b[k] = true;
    }
    void down(){
        --k;
    }
    void work(){
        bool first = true;
        if (k == (int)b.size() - 1 && sum() == s){
            cout << s << '=';
            for (int i = 0; i <= k; ++i)
                if (b[i]){
                    if (not first)
                        cout << '+';
                    first = false;
                    cout << v[i];
                }
            cout << endl;
        }
    }
    void UW(){
        while (is_up())
            up();
        work();
    }
    void traverse(){
        begin_work();
        UW();
        while (is_down()){
            if (is_right()){
                right();
                UW();
            } else
                down();
        }
    }
};

int shen_3_2_1(){
    cout << "Use tree traversal to determine if s is a sum of elements of array 1..n" << endl;
    int n = 0;
    cout << "Enter n: ";
#ifdef QT_DEBUG
    n = 4;
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
    int s = mt() % (n * n / 2);
    cout << "s = " << s << endl;

    Tree_traversal T(v, s);
    T.traverse();
    return 0;
}



