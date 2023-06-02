#include <iostream>
#include <sstream>
#include <vector>
#include <set>

using namespace std;

int main()
{
    string line;
    int c;
    set<int> s;

    getline(cin, line);
    stringstream ss(line);
    while(ss.good()){
        ss >> c;
        s.insert(c);
    }
    cout << s.size();
    return 0;
}
