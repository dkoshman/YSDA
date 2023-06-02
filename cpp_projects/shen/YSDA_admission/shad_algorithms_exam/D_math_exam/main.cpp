#include <iostream>
#include <vector>


using namespace std;

int main()
{
    vector<int> a, b;
    int n = 0, x = 0;
    cin >> n;
    a.push_back(0);
    for (int i = 0; i < n; ++i){
        cin >> x;
        a.push_back(a[i] + x);
    }
    b.push_back(0);
    for (int i = 0; i < n; ++i){
        cin >> x;
        b.push_back(b[i] + x);
    }
    int count = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n + 1; ++j)
            if (a[j] - a[i] == b[j] - b[i])
                ++count;
    cout << count << endl;

    return 0;
}
