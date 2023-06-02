/*Дан массив вещественных чисел A[1:n]
. Предложите алгоритм, находящий для каждого элемента A индекс ближайшего справа элемента,
большего его хотя бы в два раза. Если такого элемента нет, то должно возвращаться значение None.*/
#include <iostream>
#include <vector>
#include <random>
#include <ctime>

using namespace std;

int main()
{
    int n = 0;
    cin >> n;
    mt19937 mt(time(nullptr));
    vector<int> A;
    for (int i = 0; i < n; ++i)
        A.push_back(mt() % 100);
    cout << endl;
    vector<int> B, B_i, R;
    for (int i = n - 1; i >= 0; --i){
        int x = 2 * A[i];
        int ind = -1;
        for (int j = 0; j < B.size(); ++j){
            if (B[j] > x){
                ind = j;
                break;
            }
        }
        if (ind == -1){
            B.clear();
            B_i.clear();
            B.push_back(A[i]);
            B_i.push_back(i);
            R.push_back(ind);
        } else {
            R.push_back(B_i[ind]);
            B.insert(B.begin() + ind, A[i]);
            B.erase(B.begin(), B.begin() + ind);
            B_i.insert(B_i.begin() + ind, i);
            B_i.erase(B_i.begin(), B_i.begin() + ind);
        }
    }
    reverse(R.begin(), R.end());
    for (int i = 0; i < n; ++i){
        if (R[i] != -1)
            assert(A[R[i]] > 2 * A[i]);
    }
    return 0;
}
