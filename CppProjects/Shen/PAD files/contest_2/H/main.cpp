#include <iostream>


std::string common_suffix(const std::string& a, const std::string& b){
    std::string suffix = "";
    int size = (a.size() - b.size() > 0) ? b.size() : a.size();

    for (int i = 0; i < size; ++i){
        if (a[a.size() - i - 1] == b[b.size() - i - 1]){
            suffix.insert(suffix.begin(), a[a.size() - i - 1]);
        } else {
            break;
        }
    }
    return suffix;
}

