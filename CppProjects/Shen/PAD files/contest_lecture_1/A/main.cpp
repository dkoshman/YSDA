#include <iostream>
#include <vector>
#include <sstream>


std::vector<char> readLineOfChars(std::istream& in){
    std::vector<char> result;
    std::string str;
    char c = 0;

    std::getline(in, str);
    std::stringstream ss(str);
    ss >> c;
    while(ss.good()){
        result.push_back(c);
        ss >> c;
    }
    return result;
}

int findSubstring(std::vector<char> v, std::string word){
    bool sub = false;

    for (size_t i = 0; i < v.size(); ++i){
        sub = true;
        for (size_t j = 0; j < word.size(); ++j){
            if ((i + j >= v.size()) || (v[i + j] != word[j])){
                sub = false;
                break;
            }
        }
        if (sub){
            return i;
        }
    }
    return -1;
}

int main()
{
    std::vector<char> lineChars;
    std::string word;

    lineChars = readLineOfChars(std::cin);
    std::cin >> word;
    std::cout << findSubstring(lineChars, word);
    return 0;
}
