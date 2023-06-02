#include <iostream>

using namespace std;

int main()
{
    int n = 0;
    int s = 0;

    cin >> n;
    s = n % 10;
    while (n >= 10){
        n /= 10;
        s += n % 10;
    }
    cout << s;
    return 0;
}
