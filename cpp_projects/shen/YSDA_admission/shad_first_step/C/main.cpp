#include <iostream>
#include <cmath>

using namespace std;

int main()
{
    int n = 0, result = 0;

    cin >> n;
    for (int i1 = 0; i1 < n; ++i1){
        for (int j1 = 0; j1 < n; ++j1){
            for (int i2 = 0; i2 < n; ++i2){
                for (int j2 = 0; j2 < n; ++j2){
                    for (int i3 = 0; i3 < n; ++i3){
                        for (int j3 = 0; j3 < n; ++j3){
                            if (abs(i1-i2) != abs(j1-j2)
                                && abs(i1-i3) != abs(j1-j3)
                                && abs(i3-i2) != abs(j3-j2))
                                ++result;
                        }
                    }
                }
            }
        }
    }
    cout << (result / 6) << endl;
    return 0;
}
