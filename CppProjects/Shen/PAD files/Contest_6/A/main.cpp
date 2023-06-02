#include <iostream>

using namespace std;

int main()
{
    int a=0, b=0;
    cin >> a >> b;
    cout << (a ^ (1 << b));
    return 0;
}
