#include <iostream>

using namespace std;

int main()
{
    int k = 0;
    int n = 0;
    int k_mod_10 = 1;
    string s = "";

    cin >> k >> s;
    while (s.size() > 0){
        n = (n + k_mod_10 * (s.back() - '0')) % k;
        k_mod_10 = k_mod_10 * 10 % k;
        s.pop_back();
    }
    cout << n;
    return 0;
}
