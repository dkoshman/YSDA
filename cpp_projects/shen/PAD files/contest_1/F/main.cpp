#include <iostream>

using namespace std;

int main()
{
    int n = 0;
    int m = 0;

    cin >> n >> m;
    for (int i = 1; i <= n; ++i){
        cout << '\t' << i;
    }
    cout << '\n';
    for (int i = 1; i <= n; ++i){
        cout << i;
        for (int j = 1; j <= n; ++j){
            cout << '\t' << i * j % m;
        }
        cout << '\n';
    }
    return 0;
}
