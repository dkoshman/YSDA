#include <iostream>

using namespace std;

int main()
{
    int a=0, b=0, c=0;
    cin >> a >> b;
    c = a >> b << b;
    cout << (a ^ c);
    return 0;
}
