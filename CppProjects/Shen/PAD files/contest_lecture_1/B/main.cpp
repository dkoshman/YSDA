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

void findAndRevSubstring(std::vector<char> v, std::string word){
    bool sub = false;
    int id = -1;

    for (size_t i = 0; i < v.size(); ++i){
        sub = true;
        for (size_t j = 0; j < word.size(); ++j){
            if ((i + j >= v.size()) || (v[i + j] != word[j])){
                sub = false;
                break;
            }
        }
        if (sub){
            id = i;
            break;
        }
    }
    if (id != -1){
        for (size_t i = 0; i < v.size(); ++i){
            if (i < id || i > id + word.size() - 1){
                std::cout << v[i] << ' ';
            } else {
                std::cout << word[word.size() - i + id - 1] << ' ';
            }
        }
    }
}

int main()
{
    std::vector<char> lineChars;
    std::string word;

    lineChars = readLineOfChars(std::cin);
    std::cin >> word;
    findAndRevSubstring(lineChars, word);
    return 0;
}
