/*У Васи есть набор строк.
Он решил их выписать в определенном порядке:
1)определить строки минимальной длины, которые еще не выписывались
2)среди строк с минимальной длиной, которые не выписывались, выбрать минимальную лексикографически
3)выписать выбранную строку, перейти к первому пункту, если еще есть не выписанные строки
Определите, в каком порядке будут выписаны строки Васей*/
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

int main()
{
    vector<vector<string>> v;
    int n;

    cin >> n;
    v.resize(20);
    for (int i = 0; i < n; ++i){
        string s;
        cin >> s;
        v[s.size() - 1].push_back(s);
    }
    for (vector<string> &i : v){
        sort(i.begin(), i.end());
        for (string &s : i)
            cout << s << endl;
    }
    return 0;
}
