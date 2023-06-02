/*Палиндром – это строка, которая одинаково читается как слева направо, так и справа налево.
Дан набор из n строк (1 ≤ n ≤ 105). Найдите среди них такие пары, которые при конкатенации
дают палиндром. Более формально, найдите все пары (i, j) такие, что i ≠ j и строка si+sj является палиндромом.
Выведите все упорядоченные пары индексов (нумерация с единицы).
Формат ввода
В первой строке дано целое число n (1 ≤ n ≤ 100 000) — количество строк.
Далее в n строках записано по одному слову. Длина каждого слова от 1 до 10.
Слова состоят из маленьких букв английского алфавита*/

#include <iostream>
#include <string>
#include <vector>
#include <sstream>


using namespace std;

bool is_palindrom(string const &a){
    for (size_t i = 0; i <= a.size() / 2; ++i)
        if (a[i] != a[a.size() - 1 - i])
            return false;
    return true;
}

int main()
{
    int n = 0;
    cin >> n;
    vector<string> v;
    vector<int> size;
    vector<vector<bool>> straight, rev;
    straight.resize(n);
    rev.resize(n);
    string s;
    getline(cin, s);
    for (int i = 0; i < n; ++i){
        getline(cin, s);
        v.push_back(s);
        int t = s.size();
        size.push_back(t);
        string s0 = "";
        for (int j = 0; j < t; ++j){
            s0 += s[j];
            if (is_palindrom(s0))
                straight[i].push_back(true);
            else
                straight[i].push_back(false);
        }
        s0 = "";
        for (int j = t - 1; j >= 0; --j){
            s0 += s[j];
            if (is_palindrom(s0))
                rev[i].push_back(true);
            else
                rev[i].push_back(false);
        }
    }
//    for (auto &i : straight){
//        for (auto j : i)
//            cout << j << ' ';
//        cout << endl;
//    }
//    cout << endl;
//    for (auto &i : rev){
//        for (auto j : i)
//            cout << j << ' ';
//        cout << endl;
//    }
//    cout << endl;
//    for (string s : v)
//        cout << s << endl;
    bool yes = false;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j){
            if (i == j)
                continue;
            int i_s = size[i], j_s = size[j];
            if (i_s > j_s){
                if (not rev[i][i_s - j_s - 1])
                    continue;
                string v_i = v[i];
                string v_j = v[j];
                yes = true;
                for (int p = 0; p < j_s; ++p)
                    if (v_j[j_s - 1 - p] != v_i[p]){
                        yes = false;
                        break;
                    }
                if (yes)
                    cout << i + 1 << ' ' << j + 1 << endl;
                continue;
            }
            if (i_s < j_s){
                if (not straight[j][j_s - i_s - 1])
                    continue;
                string v_i = v[i];
                string v_j = v[j];
                yes = true;
                for (int p = 0; p < i_s; ++p)
                    if (v_i[p] != v_j[j_s - 1 - p]){
                        yes = false;
                        break;
                    }
                if (yes)
                    cout << i + 1 << ' ' << j + 1 << endl;
                continue;
            }
            bool yes = true;
            string v_i = v[i];
            string v_j = v[j];
            for (int p = 0; p < i_s; ++p)
                if (v_i[p] != v_j[i_s - 1 - p]){
                    yes = false;
                    break;
                }
            if (yes)
                cout << i + 1 << ' ' << j + 1 << endl;
        }

    return 0;
}
