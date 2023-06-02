#include <iostream>
#include <sstream>

using namespace std;

int main()
{
    stringstream ss;
    int n = 32;

    ss << uppercase << hex;
    for (int i = 0; i < 16; ++i){
        ss << '\t' << i;
    }
    while (n < 128){
        ss << uppercase << hex;
        ss << '\n' << n / 16;
        ss << nouppercase << dec;
        for (int i = n / 16; i == n / 16; ++n){
            ss << '\t' << char(n);
        }
    }
    cout << ss.str() << endl;
    return 0;
}
