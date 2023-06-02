#include <iostream>
#include <vector>


int fibbonachi(){
    int n = 0;
    std::vector<std::vector<int>> m0 = {{1, 1},
                                        {1, 0}};
    std::vector<std::vector<int>> m1 = m0;
    std::vector<std::vector<int>> m = m0;

    std::cout << "Fibbonachi number: ";
    std::cin >> n;
    while (n > 0){
        if (n % 2) {
            m1[0][0] = m0[0][0] * m[0][0] + m0[0][1] * m[1][0];
            m1[0][1] = m0[0][0] * m[1][0] + m0[0][1] * m[1][1];
            m1[1][0] = m0[1][0] * m[0][0] + m0[1][1] * m[1][0];
            m1[1][1] = m0[1][0] * m[1][0] + m0[1][1] * m[1][1];
            m = m1;
            --n;
        } else {
            m1[0][0] = m[0][0] * m[0][0] + m[0][1] * m[1][0];
            m1[0][1] = m[0][0] * m[1][0] + m[0][1] * m[1][1];
            m1[1][0] = m[1][0] * m[0][0] + m[1][1] * m[1][0];
            m1[1][1] = m[1][0] * m[1][0] + m[1][1] * m[1][1];
            m = m1;
            n /= 2;
        }
    }
    std::cout << m[1][1] << std::endl;
    return m[1][1];
}
