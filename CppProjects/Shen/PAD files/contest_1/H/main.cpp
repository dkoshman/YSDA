#include <iostream>

using namespace std;

int main()
{
    int n = 0;
    int a = 0;
    int r[20000];

    cin >> n;
    for (int i = 1; i <= n; i++){
        cin >> a;
        r[a] = i;
    }
    for (int i = 1; i <= n; i++){
        cout << r[i] << ' ';
    }
    return 0;
}
