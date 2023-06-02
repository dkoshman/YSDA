#include <iostream>

using namespace std;

int main()
{
    int d = 0;
    int m = 0;
    int y = 0;
    int months[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    bool v = false;

    cin >> d >> m >> y;
    v = (y % 400 == 0) || ((y % 4 == 0) && (y % 100 != 0));
    if (v){
        months[1] += 1;
    }
    d += 2;
    if (d > months[m - 1]){
        d %= months[m - 1];
        m += 1;
    }
    if (m > 12) {
        y += 1;
        m %= 12;
    }
    cout << d << ' ' << m << ' ' << y;
    return 0;
}
