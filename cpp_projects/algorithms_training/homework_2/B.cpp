#include <iostream>
#include <string>
#include <assert.h>

std::string GetSmallestTail(const std::string& x_message, const std::string& z_message) {
    std::string y_message(z_message);
    for (size_t offset = 0; offset < x_message.size(); ++offset) {
        for (size_t z_pos = 0; z_pos < z_message.size(); ++z_pos) {
            if (x_message[(offset + z_pos) % x_message.size()] != z_message[z_pos]) {
                break;
            }
            if ((offset + z_pos + 1) % x_message.size() == 0 &&
                z_message.size() - z_pos - 1 < y_message.size()) {
                y_message = z_message.substr(z_pos + 1);
            }
        }
    }
    return y_message;
}

void Test() {
    assert(GetSmallestTail("mama", "amamam") == "m");
    assert(GetSmallestTail("computer", "comp") == "comp");
    assert(GetSmallestTail("ejudge", "judge").empty());
    assert(GetSmallestTail("judge", "judge").empty());
    assert(GetSmallestTail("judge", "judgejudg") == "judg");
    assert(GetSmallestTail("mmm", "m").empty());
    assert(GetSmallestTail("ilo", "iloiloailoa") == "ailoa");
}

int main() {
    Test();
    std::string x_message, z_message;
    std::cin >> x_message >> z_message;
    std::cout << GetSmallestTail(x_message, z_message) << std::endl;
    return 0;
}
