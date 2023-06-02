#include <iostream>
#include <set>

using namespace std;

int main()
{
    string line;
    size_t pt;
    set<string> s;

    while(cin.good()){
        line.clear();
        getline(cin, line);
//        if (line.size() == 0)
//            break;
        pt = line.find_last_of('/');
        while (line.size() && (pt != string::npos)){
            line = line.substr(0, pt + 1);
            s.insert(line);
            line.pop_back();
            pt = line.find_last_of('/');
        }
    }
    for (string i : s){
        cout << i << endl;
    }
    return 0;
}
