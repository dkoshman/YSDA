#include <iostream>
#include <vector>

using namespace std;

int main()
{
    int n = 0, a = 0;
    bool first = false;
    vector<string> v;

    cin >> n;
    v.resize(n);
    for (int i = 0; i < n; ++i)
        cin >> v[i];
    for (int i = 0; i < 1 << n; ++i){
        cout << '[';
        a = i;
        first = true;
        for (int j = 0; j < n; ++j){
            if (a % 2){
                if (not first){
                    cout << ',';
                }
                cout << v[j];
                first = false;
            }
            a >>= 1;
        }
        cout << ']' << endl;
    }
    return 0;
}
