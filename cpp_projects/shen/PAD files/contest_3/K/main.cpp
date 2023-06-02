#include <iostream>
#include <unordered_set>

using namespace std;

int main()
{
    char op;
    unordered_set<string> dict;
    unordered_set<string>::iterator it;
    string s;

    cin >> op;
    while (op != '#'){
        cin >> s;
        if (op == '+'){
            dict.insert(s);
        } else if (op == '-'){
            dict.erase(s);
        } else if (op == '?'){
            it = dict.find(s);
            cout << (it == dict.end() ? "NO" : "YES") << endl;
        }
        cin >> op;
    }
    return 0;
}
