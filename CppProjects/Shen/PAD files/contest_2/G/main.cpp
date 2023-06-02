#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main()
{
    int n = 0;
    int m = 0;
    int k = 0;
    vector<vector<int>> field;
    vector<vector<int>> repr;
    vector<int> line;
    int a = 0;
    int b = 0;

    cin >> n >> m >> k;
    field.resize(n);
    for (int i = 0; i < n; ++i){
        field[i].resize(m);
        fill(field[i].begin(), field[i].end(), 0);
    }
    repr.resize(n + 2);
    for (int i = 0; i < n + 2; ++i){
        repr[i].resize(m + 2);
        fill(repr[i].begin(), repr[i].end(), 0);
    }
    for (int i = 0; i < k; ++i){
        cin >> a >> b;
        field[a - 1][b - 1] = 1;
    }
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < m; ++j){
            if (field[i][j] == 1){
                ++repr[i - 1 + 1][j - 1 + 1];
                ++repr[i - 1 + 1][j - 0 + 1];
                ++repr[i - 1 + 1][j + 1 + 1];
                ++repr[i - 0 + 1][j - 1 + 1];
                ++repr[i - 0 + 1][j + 1 + 1];
                ++repr[i + 1 + 1][j - 1 + 1];
                ++repr[i + 1 + 1][j - 0 + 1];
                ++repr[i + 1 + 1][j + 1 + 1];
            }
        }
    }
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < m; ++j){
            if (field[i][j] == 1){
                cout << "* ";
            } else {
                cout << repr[i + 1][j + 1] << ' ';
            }
        }
        cout << endl;
    }
    return 0;
}
