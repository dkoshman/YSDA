#include <iostream>

using namespace std;

int main()
{
    int d0 = 0;
    int h0 = 0;
    int m0 = 0;
    int d1 = 0;
    int h1 = 0;
    int m1 = 0;
    int t1 = 24 * 60 * 7;
    int week_m = t1;
    int t0 = 0;
    int dt = 0;
    int d = 0;
    int h = 0;
    int m = 0;
    int t = 0;
    int n = 0;

    cin >> d0 >> h0 >> m0 >> n;
    t0 = 24 * 60 * (d0 - 1) + 60 * h0 + m0;
    for (int i = 0; i < n; i++){
        cin >> d >> h >> m;
        if (d == 0){
            if ((h > h0) || ((h == h0) && (m >= m0))){
                d = d0;
            } else {
                d = (d0 % 7) + 1;
            }
        }
        t = 24 * 60 * (d - 1) + 60 * h + m;
        dt = (week_m + t - t0) % week_m;
        if (dt < t1){
            t1 = dt;
            d1 = d;
            h1 = h;
            m1 = m;
        }

    }
    cout << d1 << ' ' << h1 << ' ' << m1;
    return 0;
}
