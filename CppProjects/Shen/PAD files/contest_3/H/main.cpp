#include <iostream>
#include <sstream>
#include <vector>
#include <map>

using namespace std;

int main()
{
    string line;
    long double c;
    map<long double, bool> s;

    getline(cin, line);
    stringstream ss(line);
    ss >> ws;
    while(ss.good()){
        ss >> c;
        cout << (s[c] ? "YES" : "NO") << endl;
        s[c] = true;
        ss >> ws;
    }
    return 0;
}
