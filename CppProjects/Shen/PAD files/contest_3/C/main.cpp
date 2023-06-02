#include <iostream>
#include <vector>

using namespace std;

int main()
{
    int n = 0;
    int a = 0;
    std::vector<std::vector<int>> m;
    bool associative = true;

    cin >> n;
    m.resize(n);
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            cin >> a;
            m[i].push_back(a);
        }
    }
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            for (int k = 0; k < n; ++k){
                if (m[m[i][j]][k] != m[i][m[j][k]])
                    associative = false;
            }
        }
    }
    cout << (associative ? "YES" : "NO") << endl;
    return 0;
}
