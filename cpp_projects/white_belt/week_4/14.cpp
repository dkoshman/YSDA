#include <iostream>
#include <string>

using std::string, std::runtime_error, std::cout, std::endl;

void EnsureEqual(const string& left, const string& right) {
    if (left != right) {
        throw std::runtime_error(left + " != " + right);
    }
}
//
//int main() {
//    try {
//        EnsureEqual("C++ White", "C++ White");
//        EnsureEqual("C++ White", "C++ Yellow");
//    } catch (runtime_error& e) {
//        cout << e.what() << endl;
//    }
//    return 0;
//}