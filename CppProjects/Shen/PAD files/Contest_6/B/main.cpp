#include <iostream>

using namespace std;

int main()
{
    int a=0, b=0;
    cin >> a >> b;
    cout << (a >> b) % 2;
    return 0;
}
