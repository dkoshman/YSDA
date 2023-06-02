/*Имеется доска n×n, окрашенная в белый и черный цвета в шахматном порядке, причем белых клеток не меньше, чем черных.
Посчитайте число способов разместить на доске трех шахматных слонов так, чтобы ни один из них не находился под боем другого.*/

#include <iostream>

using namespace std;

int main()
{

//    Let's numerate the officers.
//    Property a: 1st and 2nd officer hit each other
//    Property b: 2nd and 3rd officer hit each other
//    Property c: 3rd and 1st officer hit each other
//    N(~a,~b,~c) = N - N(a) - N(b) - N(c) + N(ab) + N(bc) + N(ac) - N(abc) = N - 3*N_1+ 3*N_2 - N_3
//    N_1 -- number of arrangements where a particular pair (ie 1st and 2nd officers) hits each other
//    N_2 -- number of arrangements where two particular pairs (ie 1st hits 2nd and 3rd) hit each other
//    N_3 -- number of arrangements where all three pairs hit each other
//    Then the answer is N(~a,~b,~c) / 3! as we don't distinguish between officers.
    int n = 0;
    cin >> n;
    int n_squared = n * n;
    int N = n_squared * (n_squared - 1) * (n_squared - 2);
    int N_1 = 0, N_2 = 0, N_3 = 0;

//    I split up the arrangements where 1st officer doesnt stand on the main or secondary diagonal
//    into classes of symmetry: 1st class has 1st officer standing under the main diagonal
//    and hitting 2nd officer parallel to the main diagonal. The other three cases are the three other classes.
//    Let's iterate through diagonals up to the main exclusively, i is the number of the diagonal, starting with 1.
//    There's no room for a pair of officers hitting each other on the 1st diagonal, so we skip it.
    for (int i = 2; i < n; ++i){
//      i spaces on the diagonal for 1st officer, i-1 for 2nd and any other place for 3rd
        N_1 += i * (i - 1) * (n_squared - 2);
//      i spaces on the diagonal for 1st officer, i-1 for 2nd and i-2 for 3rd
        N_3 += i * (i - 1) * (i - 2);
        for (int j = 0; j < i; ++j){
//          (i,j) - note it's not orthogonal coordinaters - is the place for 1st officer.
//          There are i-1 places left on the diagonal for 2nd officer
//          and any of the remaining places on the diagonals passing through (i,j) for 3rd.
            N_2 += (i - 1) * (n - 2 + min(2 * j, 2 * (i - j - 1)));
        }
    }
//    due to 4-way symmetry of the chess field except for main diagonal
    N_1 *= 4;
    N_2 *= 4;
    N_3 *= 4;
//    due to 2-way symmetry of the diagonal
    N_1 += 2 * (n * (n - 1) * (n_squared - 2));
    N_3 += 2 * n * (n - 1) * (n - 2);
    for (int j = 0; j < n; ++j){
        N_2 += 2 * (n - 1) * (n - 2 + min(2 * j, 2 * (n - j - 1)));
    }
    cout << "N=" << N / 6 << ", N_1=" << N_1 / 2 << ", N_2=" << N_2 / 2 << ", N_3=" << N_3 / 6 << endl;
    cout << "There are " << (N - 3*N_1 + 3*N_2 - N_3) / 6  << " ways to arrange 3 officers on a "
         << n << " by " << n << " chess field so that no pair of officers hit each other."<< endl;

    return 0;
}
