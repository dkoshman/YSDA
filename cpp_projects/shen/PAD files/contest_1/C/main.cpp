#include <iostream>

using namespace std;

int main()
{
    int v = 0;
    int t = 0;
    const int MKAD = 109;

    cin >> v >> t;
    cout << ((v * t) % MKAD + MKAD) % MKAD;
    return 0;
}
