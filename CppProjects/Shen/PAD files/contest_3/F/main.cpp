#include <iostream>
#include <map>
#include <vector>

using namespace std;

int main()
{
    int n = 0;
    map<string, pair<long long, long long>> grades;
    string name;
    long long g = 0;

    cin >> n;
    for (int line = 0; line < n; ++line){
        cin >> name >> g;
        ++grades[name].first;
        grades[name].second += g;
        cout << (long long)(grades[name].second / grades[name].first) << endl;
    }
    return 0;
}
