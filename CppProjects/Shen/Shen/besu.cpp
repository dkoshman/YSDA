#include <iostream>


int besu() {
    int a[100], n=0, x, v[100];
    std::string inp="0";
    std::cout << "Enter polynomial, q to quit:\n";
    while (inp != "q"){
        a[n] = stoi(inp);
        std::cout << "x^" << n << ": ";
        std::cin >> inp;
        n++;
    }
    n--;
    std::cout << "At what point to calculate its value?\nx=";
    std::cin >> x;
    std::cout << "Besu algorithm: v_n = a_n + x * v_n-1; v_0 = a_0\n";
    for (int i=n; i>0; --i){
        std::cout << a[i] << '\t';
    }
    v[n] = a[n];
    std::cout << '\n' << v[n] << '\t';
    for (int i=n-1; i>0; --i){
        v[i] = v[i+1] * x + a[i];
        std::cout <<  v[i] << '\t';
    }
    std::cout << "\nP(" << x << ") = " << v[1] << "\n";
    return 0;
}
