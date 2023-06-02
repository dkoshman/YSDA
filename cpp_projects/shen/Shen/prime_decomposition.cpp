#include <iostream>


int prime_decomposition(){
    int n = 0;
    int m = 0;
    int n0 = 0;
    int m0 = 0;
    int n_tmp = 0;
    int square = 2;

    std::cout << "Enter complex number to decompose: ";
    std::cin >> n >> m; // n+im
    std::cout << n << "+i" << m << "=";
    while (n*n + m*m > 1){
        n0 = 1;
        m0 = 0;
        while (true){
            if ((n0*n0 + m0*m0 > 1)
               && (n0*n0 + m0*m0 <= square)
               && (((n*n0 + m*m0) % (n0*n0 + m0*m0) == 0)
               && (((-n*m0 + m*n0) % (n0*n0 + m0*m0) == 0)))){
                break;
            }
            if (m0 < 0 && n0 == 0){
                n0 = 1 - m0;
                m0 = 0;
                if (n0 > square){
                    n0 = 1;
                    m0 = 0;
                    ++square;
                }
            } else {
                if (m0 <= 0){
                    --n0;
                    m0 = 1 - m0;
                } else {
                    m0 = -m0;
                }
            }
        }
        n_tmp = (n*n0 + m*m0) / (n0*n0 + m0*m0);
        m = (-n*m0 + m*n0) / (n0*n0 + m0*m0);
        n = n_tmp;
        std::cout << '(' << n0 << "+i" << m0 << ')' << '*';
    }
    std::cout << '(' << n << "+i" << m << ')' << std::endl;
    return 0;
}
