#include <sstream>
#include <vector>
#include <iostream>


int shen_1_1_32(){
    std::cout << "Enter the function 1..n to 1..n:" << std::endl;
    std::string s;
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(std::cin, s);
    std::stringstream ss(s);
    std::vector<int> v;
    int x = 0;
    while (ss.good()){
        ss >> x;
        v.push_back(x);
    }
    for (int i : v){
        if (i > v.size()){
            std::cout << "This is not a transposition" << std::endl;
        }
    }
    int a = v[0];
    int b = v[v[0] - 1];
    // invariant: a = f^k(1), b = f^{2k}(1), k = 1, 2, ...
    while (a != b){
        a = v[a - 1];
        b = v[v[b - 1] - 1];
    }
    int p = 1;
    b = v[a - 1];
    while (a != b){
        b = v[b - 1];
        ++p;
    }
    std::cout << "Period of 1, f(1), f(f(1)), ... is " << p << std::endl;
    return 0;
}
