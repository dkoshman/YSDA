#include <iostream>


int euclidian(){
    int r[100], q[100], n, u[100], v[100];
    n = 1;
    u[0] = v[1] = 1;
    u[1] = v[0] = 0;

    std::cout << "Two numbers to find their gcd:\n";
    std::cin >> r[0] >> r[1];

    while (r[n] > 0){
        q[n] = r[n-1] / r[n];
        r[n+1] = r[n-1] - r[n] * q[n];
        std::cout << '\n' << r[n-1] << "=" << r[n] << "*" << q[n] << "+" << r[n+1];
        u[n+1] = u[n-1] - u[n] * q[n];
        v[n+1] = v[n-1] - v[n] * q[n];
        ++n;
    }
    std::cout << "\ngcd of " << r[0] << " and " <<  r[1] << " is " << r[n-1]
              << ", their lcd is " << r[0] * r[1] / r[n-1] << '\n';
    std::cout << u[n-1] << '*' << r[0] << '+' << v[n-1] << '*' << r[1] << '='
                        << u[n-1] * r[0] + v[n-1] * r[1] << '\n';
    return r[n-1];
}
