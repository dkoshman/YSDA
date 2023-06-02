#include <iostream>


int divmod()
{
    int a, b, q, a_0;
    q = 0;
    std::cout << "Two numbers to be divided:";
    std::cin >> a >> b;
    a_0 = a;
    while (a >= b){
        a -= b;
        q += 1;
    }
    std::cout << '\n' << a_0 << " = " << b << "*" << q << " + " << a << '\n';
    return 0;
}
