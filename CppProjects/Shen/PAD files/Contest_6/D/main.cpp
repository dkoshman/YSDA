#include <iostream>

using namespace std;

int main()
{
    unsigned long long a=0;
    int b=0;
    cin >> a;
    while (a > 0){
        b += a % 2;
        a >>= 1;
    }
    cout << b;
    return 0;
}
