#include <iostream>


std::string ExtractDigits(const std::string& s){
    std::string result;

    for (char c : s){
        if (std::isdigit(c)){
            result.push_back(c);
        }
    }
    return result;
}
