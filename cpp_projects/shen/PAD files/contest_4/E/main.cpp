#include <iostream>
#include <vector>

using namespace std;

int word2vec(vector<int> v, vector<int> w){
    int res = 0;
    for (size_t i = 0; i < v.size(); ++i)
        res += v[i] * w[i];
    return res;
}

int main()
{
    int m = 0;
    int n = 0;
    int r = 0;
    vector<vector<int>> v;
    vector<string> w;
    vector<int> res;

    cin >> m >> n;
    v.resize(m);
    w.resize(m);
    res.resize(m);
    for (int i = 0; i < m; ++i){
        cin >> w[i];
        v[i].resize(n);
        for (int j = 0; j < n; ++j)
            cin >> v[i][j];
        r = word2vec(v[i], v[0]);
        if (i == 0)
            continue;
        if (i == 1){
            res.clear();
            res.push_back(r);
            res.push_back(i);
        } else {
            if (r == res[0])
                res.push_back(i);
            else if (r > res[0]){
                res.clear();
                res.push_back(r);
                res.push_back(i);
            }

        }
    }
    for (size_t i = 1; i < res.size(); ++i){
        cout << w[res[i]] << endl;
    }
    return 0;
}
