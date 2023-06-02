#include <iostream>
#include <bitset>

using namespace std;


int main()
{
    string s;
    int size = 0;
    bitset<8> a;
    getline(cin, s);

    for (std::size_t i = 0; i < s.size(); ++i)
    {
        a = bitset<8>(s.c_str()[i]);
        if (a[7] == 1){
            if (a[5] == 1){
                if (a[4] == 1){
                    ++i;
                }
                ++i;
            }
            ++i;
        }
        ++size;
    }
    cout << size;
    return 0;
}
