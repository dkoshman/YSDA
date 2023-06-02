#include <iostream>
#include <map>
#include <algorithm>

using namespace std;

int main()
{
    int n = 0;
    int k = 0;
    int x = 0;
    multimap<int, int> m;

    cin >> n >> k;
    for (int i = 0; i < k; ++i){
        cin >> x;
        m.insert(pair<int, int> (x, i));
    }
    cout << m.begin()->first << endl;
    for (int i = k; i < n; ++i){
        cin >> x;
        m.insert(pair<int, int> (x, i));
        while (m.begin()->second < i - k + 1){
            m.erase(m.begin());
        }
        cout << m.begin()->first << endl;
    }
    return 0;
}

