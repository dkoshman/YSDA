#include <iostream>

using namespace std;

int main()
{
    int n = 0;
    double s = 0;

    cin >> n;
    for (int i = 1; i <= n; ++i){
        s += (2.0 * (i % 2) - 1) / i;
    }
    cout << s;
    return 0;
}
