#include <vector>
#include <string>
#include <iostream>


int* chinese_euclidian(int, int);

int chinese (){
    std::string tmp_1 = "";
    std::string tmp_2 = "";
    int size = 0;
    int M = 0;
    int A = 0;
    int a[100], m[100];
    int* res;

    std::cout << "q to exit" << std::endl;
    while (true){
        std::cout << "x = <remainder> (mod <factor>) ";
        std::cin >> tmp_1 >> tmp_2;
        if (tmp_1 == "q" || tmp_2 == "q") {break;}
        m[size] = stoi(tmp_2);
        if (m[size] == 0){
            std::cout << "Error: factors can't be zero\n";
            return 1;
        }
        a[size++] = stoi(tmp_1) % m[size];
    }
    for (int i = 0; i < size; ++i){
        for (int j = i + 1; j < size; ++j){
            res = chinese_euclidian(m[i], m[j]);
            if (*(res) != 1){
                std::cout << "Error: Not all factors are pairwise coprime\n";
                return 1;
            }
        }
    }
    M = m[0];
    A = a[0];
    for (int i = 1; i < size; ++i){
        res = chinese_euclidian(M, m[i]);
        A = (*(res + 1)) * M * a[i] + (*(res + 2)) * m[i] * A;
        M *= m[i];
        A = (A + M) % M;
    }
    std::cout << "x = " << A << std::endl;
    return 0;
}


int* chinese_euclidian(int x, int y){
    int r[100], q[100], n, u[100], v[100];
    static int result[3];
    n = 1;
    u[0] = v[1] = 1;
    u[1] = v[0] = 0;
    r[0] = x;
    r[1] = y;

    while (r[n] > 0){
        q[n] = r[n-1] / r[n];
        r[n+1] = r[n-1] - r[n] * q[n];
        u[n+1] = u[n-1] - u[n] * q[n];
        v[n+1] = v[n-1] - v[n] * q[n];
        ++n;
    }

    result[0] = r[n-1];
    result[1] = u[n-1];
    result[2] = v[n-1];
    return result;
}

