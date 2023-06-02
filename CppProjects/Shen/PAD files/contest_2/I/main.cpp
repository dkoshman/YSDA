#include <iostream>
#include <vector>

std::vector<std::string> split(const std::string& str, char delimiter){
    std::vector<std::string> result;
    size_t a = 0;
    std::string buffer;

    for (size_t i = 0; i < str.size(); ++i){
        if (str[i] == delimiter){
            result.push_back(str.substr(a, i - a));
            a = i + 1;
        }
    }
    result.push_back(str.substr(a));
    return result;
}
