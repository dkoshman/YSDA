#include <iostream>


int shen_1_1_33(){
    int n = 0;
    std::cout << "At what point to calculate f(n)? "
                 "f(2n + 1) = f(n) + f(n + 1), f(2n) = f(n), f(0) = 0, f(1) = 1" << std::endl;
    std::cin >> n;
    int f0 = 0, f1 = 1;
//    f(2n) = f(n), f(2n + 1) = f(n) + f(n + 1)
    int a = 1, b = 0, k = n;
//    Invariant: f(n) = a * f(k) + b * f(k + 1)
    while (k > 0) {
        if (k % 2 == 0){
            k /= 2;
//            f(n) = a * f(2k) + b * f(2k + 1) = (a + b) * f(k) + b * f(k + 1)
            a += b;
        } else {
            k /= 2;
//            f(n) = a * (f(k) + f(k + 1)) + b * f(k + 1)
            b += a;
        }
    }
    std::cout << (f0 * a + f1 * b) << std::endl;
    return 0;
}
