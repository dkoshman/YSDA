#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <functional>

#include "quadrature.h"


int main()
{
    // Creating an alias for a type of function like cos(), sin() etc.
    // The datatype is double F(double), but it's not a function pointer!
    // Consider using std::function with them.
//    using F = decltype (cos);
    using F = std::function<double(double)>;

    std::string input;
    std::cin >> input;

    std::unique_ptr<IntegrationMethod<F>> method;
    if (input == "rectangle")
        method.reset(new RectangleRule<F>);
    else
        method.reset(new TrapezoidalRule<F>);

    double x, y;
    std::cin >> x >> y;

    int n;
    std::cin >> n;
    F cos_f = [](double z){return cos(z);};
    F sin_f = [](double z){return sin(z);};
    std::cout << method->integrate(cos_f, x, y, n) << "\n";
    std::cout << method->integrate(sin_f, x, y, n) << "\n";
}
