#include <iostream>
#include <vector>

using namespace std;

int count (vector<int> v, int a, int m){
    if (v.size() == 0){
        return a == ((1 << m) - 1);
    }
    int b = 0;
    vector<int> u = v;
    b = u[u.size() - 1];
    u.pop_back();
    return count(u, a, m) + count(u, a | b, m);
}

int main()
{
    int n, m, b, x;
    vector<int> v;
    cin >> n >> m;
    for (int i = 0; i < n; ++i){
        x = 0;
        for (int j = 0; j < m; ++j){
            cin >> b;
            x = (x << 1) + b;
        }
        v.push_back(x);
    }
    cout << count(v, 0, m);

    return 0;
}
