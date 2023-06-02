#include <iostream>

using namespace std;

void pbin(unsigned long long a){
    string s = to_string(a % 2);
    while(a > 1){
        a >>= 1;
        s = to_string(a % 2) + s;
    }
    cout << s << endl;
}

int main()
{
    unsigned long long int a = 0, n = 0, b = 0, c = 0, max = 0;
    cin >> a;
    c = a;
    while (c > 0){
        ++n;
        c >>= 1;
    }
    a += a << n;
    b = (1 << n) - 1;
    for (int i = 0; i < n; ++i){
        c = a & b;
        if (c > max)
            max = c;
        a >>= 1;
    }
    cout << max;
    return 0;
}
