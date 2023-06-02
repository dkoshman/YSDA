#include <iostream>
#include <vector>


std::string join(const std::vector<std::string>& tokens, char delimiter){
    std::string result;

    for (std::string str : tokens){
        result.append(str);
        result.push_back(delimiter);
    }
    if (tokens.size() > 0){
        result.pop_back();
    }
    return result;
}
