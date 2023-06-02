#include <iostream>

using namespace std;

typedef unsigned long long shlong;
int main()
{
    shlong a = 1;
    shlong b = 1;
    shlong c = 1;
    shlong tmp = 0;
    int n = 0;

    cin >> n;
    for (int i = 0; i < n; ++i){
        cout << a << endl;
        tmp = a;
        a = b;
        b = c;
        c = tmp + a + b;
    }
    return 0;
}
