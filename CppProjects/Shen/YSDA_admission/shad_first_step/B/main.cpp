/*Мощностью элемента массива назовем разность максимального элемента, стоящего справа от него, и минимального элемента,
стоящего от него слева. Мощность крайних элементов будем считать равной
У вас имеется первоначально пустой массив, к которому по очереди справа приписываются n
целых чисел. На каждом шаге посчитайте и выведите наибольшее значение среди мощностей его элементов*/

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main()
{
    int n = 0, x = 0, m = 0, t = 0, change = 0;
    vector<int> minv, maxv;

    cin >> n;
    for (int i = 0; i < n; ++i){
        cin >> x;
        minv.push_back(x);
        if (i > 0)
            minv[-1] = min(minv[-2], minv[-1]);
        maxv.push_back(x);
        change = 1;
        for (int j = maxv.size() - 1; j > 0; --j){
            t = max(maxv[j - 1], maxv[j]);
            if (maxv[j - 1] >= t){
                change = max(1, j - 2);
                break;
            }
            maxv[j - 1] = t;
        }
        for (int j = change; j < i; ++j)
            m = max(m, maxv[j + 1] - minv[j - 1]);
        cout << m  << ' ' << endl;
    }
    return 0;
}
