#include <iostream>
#include <algorithm>
#include <set>

using namespace std;

int main()
{
    string line;
    set<char> r = {};
    set<char> s = {};
    set<char> intersect;
    bool first_line = true;

    while (cin.good()){
        line.clear();
        getline(cin, line);
        if (line.size() == 0){
            break;
        }
        s.clear();
        for (char c : line){
            s.insert(c);
        }
        if (first_line){
            r = s;
            first_line = false;
        } else {
            intersect.clear();
            set_intersection(r.begin(), r.end(), s.begin(), s.end(),
                             std::inserter(intersect, intersect.begin()));
            r = intersect;
        }
    }
    for (char c : r){
        cout << c;
    }
    return 0;
}
