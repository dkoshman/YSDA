#include <iostream>


int shen_1_1_35(){
    int a = 0, b = 0;
    std::cout << "Enter integers a, b to divide a by b:" << std::endl;
    std::cin >> a >> b;
    int B = b;
    while (B <= a)
        B *= 2;
    int q = 0, r = a;
//    Invariant: a = B * q + r, 0 <= r < B, B = b * (2 ^ n)
    while (B != b) {
        B /= 2;
        q *= 2;
        if (r >= B){
            r -= B;
            q += 1;
        }
    }
    std::cout << a << " = " << b << " * " << q << " + " << r << std::endl;
    return 0;
}
