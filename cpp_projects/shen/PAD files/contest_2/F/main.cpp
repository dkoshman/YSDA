#include <iostream>
#include <sstream>
#include <algorithm>

using namespace std;

int main()
{
    string n;
    bool carry_over = true;
    int i = 0;
    stringstream ss;
    string str;

    cin >> n;
    for (string::reverse_iterator it = n.rbegin(); it < n.rend(); ++it){
        i = *it - '0';
        if (carry_over){
            i = (i + 1) % 10;
            if (i != 0){
                carry_over = false;
            }
        }
        ss << i;
    }
    if (carry_over){
        ss << 1;
    }
    str = ss.str();
    reverse(str.begin(), str.end());
    cout << str;
    return 0;
}
