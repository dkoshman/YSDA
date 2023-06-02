/*Дан массив a из n целых чисел. Напишите программу, которая найдет число, которое встречается в массиве наибольшее число раз.*/

#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int main()
{
    int n = 0;
    vector<unsigned> v;

    cin >> n;
    v.resize(n);
    for (int i = 0; i < n; ++i)
        cin >> v[i];
    sort(v.begin(), v.end());
    unsigned prev = v[0];
    unsigned res = 0;
    int m_freq = 0;
    int freq = 1;
    for (int i = 1; i < n; ++i){
        if (prev == v[i]){
            ++freq;
        } else {
            if (freq == m_freq)
                res = max(res, prev);
            if (freq > m_freq){
                res = prev;
                m_freq = freq;
            }
            prev = v[i];
            freq = 1;
        }
    }
    if (freq == m_freq)
        res = max(res, prev);
    if (freq > m_freq)
        res = prev;
    cout << res << endl;
    return 0;
}
