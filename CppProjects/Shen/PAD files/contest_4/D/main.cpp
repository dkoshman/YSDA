#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void print(vector<int> v){
    for (int i : v)
        cout << i;
    cout << endl;
}

int main()
{
    int n = 0;
    pair<int, size_t> max = {0, 0};
    vector<int> v;
    vector<int> last;

    cin >> n;
    v.resize(n);
    last.resize(n);
    for (int i = 0; i < n; ++i){
        v[i] = i + 1;
        last[i] = n - i;
    }
    print(v);
    while (v != last){
        max = {v[n - 1], n - 1};
        for (int i = n - 2; i >= 0; --i){
            if (v[i] < max.first){
                for (int j = i + 1; j < n; ++j){
                    if (v[j] > v[i] && v[j] < max.first){
                        max.second = j;
                    }
                }
                swap(v[i], v[max.second]);
                sort(v.begin() + i + 1, v.end());
                break;
            }
            if (v[i] > max.first){
                max.first = v[i];
                max.second = i;
            }
        }
        print(v);
    }
    return 0;
}
