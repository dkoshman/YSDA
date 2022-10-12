#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

long long t[1000000];
int n;

long long sum(int r) {
    long long result = 0;
    for (; r >= 0; r = (r & (r + 1)) - 1) {
        result += t[r];
    }
    return result;
}

void inc(int i, long long delta) {
    for (; i < n; i = (i | (i + 1)))
        t[i] += delta;
}

int main() {
    cin >> n;
    for (int i = 0; i < n; i++) {
        long long p;
        cin >> p;
        cout << sum(p - 1) << " ";
        inc(p, p);
    }
}