#include <iostream>
#include <map>
#include <set>

using namespace std;

int main()
{
    map<char, int> dict;
    map<char, int>::iterator it;
    string line;
    int m = 0;

    while(cin.good()){
        line.clear();
        getline(cin, line);
//        if (line.size() == 0)
//            break;
        for (char c : line){
            if (c && (c != '\n') && (c != ' ')){
                it = dict.find(c);
                if (it != dict.end())
                    ++dict[c];
                else
                    dict[c] = 1;
            }
        }
    }
    for (pair<char, int> p : dict){
        m = p.second > m ? p.second : m;
    }
    for (int i = m; i > 0; --i){
        for (pair<char, int> p : dict){
            cout << (p.second >= i ? '#' : ' ');
        }
        cout << endl;
    }
    for (pair<char, int> p : dict){
        cout << p.first;
    }
    cout << endl;
    return 0;
}
