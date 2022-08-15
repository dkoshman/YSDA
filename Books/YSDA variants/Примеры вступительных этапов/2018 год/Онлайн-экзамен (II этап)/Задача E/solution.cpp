#include <iostream>
#include <set>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    long long sum = 0;
    int n;
    cin >> n;
    multiset<long long> s;
    for (int i = 0; i < n; i++) {
        long long a;
        cin >> a;
        s.insert(a);
        sum += a;
    }
    long long result = 0;
    while (s.size() > 2) {
        auto it_a = s.begin();
        auto it_b = s.end();
        it_b--;
        it_b--;

        long long x = *it_a + *it_b;
        result += x;

        sum -= *it_a;
        sum -= *it_b;
        sum += x;

        s.erase(it_a);
        s.erase(it_b);
        s.insert(x);
    }
    cout << sum + result;
    //system("pause");
    return 0;
}
