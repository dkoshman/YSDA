/*Ускорьте выполнение следующего псевдокода:
  function Foo(array_of_ints a): // входные параметры: массив целых чисел
    result = 0
    while size(a) > 2:           // пока длина массива больше двух элементов
      sort(a)                    // отсортировать массив по возрастанию элементов
      n = size(a)
      x = a[0] + a[n - 2]
      result += x                // добавить x к накапливаемому результату
      delete(a, n-2)             // удалить элемент по индексу (n-2)
      delete(a, 0)               // удалить элемент по индексу 0
      add(a, x)                  // добавить элемент со значением x в конец массива
    // end while
    return sum(a) + result       // к накапливаемому результату добавить сумму элементов*/

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using shlong = unsigned long long;

int main()
{
    int n = 0;
    shlong res = 0, tmp = 0;
    vector<unsigned> v;
    cin >> n;
    v.resize(n);
    for (int i = 0; i < n; ++i)
        cin >> v[i];
    if (n == 1){
        cout << v[0];
        return 0;
    }
    sort(v.begin(), v.end());
    shlong a = *(v.end() - 2);
    shlong b = *(v.end() - 1);
    for (int i = 0; i < n - 2; ++i){
        res += v[i] + a;
        a += v[i];
        if (a > b){
            tmp = a;
            a = b;
            b = tmp;
        }
    }
    res += a + b;
    cout << res;
    return 0;
}
