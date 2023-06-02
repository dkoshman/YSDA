#include <iostream>
#include <vector>

using namespace std;

void print(vector<int> v){
    for (size_t i = 0; i < v.size(); ++i){
        for (int j = 0; j < v[i]; ++j)
            cout << 0;
        if (i != v.size() - 1)
            cout << 1;
    }
    cout << endl;
}

int main() {
    int n = 0;
    int k = 0;
    int tmp = 0;
    vector<int> v;

    cin >> n >> k;
    v.resize(k + 1);
    v[0] = n - k;
    if (n != 0)
        print(v);
    while (v[k] != n - k){
        for (int i = k; i > 0; --i){
            if (v[i - 1] != 0){
                --v[i - 1];
                tmp = v[k];
                v[k] = 0;
                v[i] = tmp + 1;
                break;
            }
        }
        print(v);
    }
};
