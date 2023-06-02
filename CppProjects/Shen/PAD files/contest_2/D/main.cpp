#include <iostream>
#include <vector>

using namespace std;

int main()
{
    string word;
    const string silent = "aehiouwy";
    const vector<string> code = {"bfpv", "cgjkqsxz", "dt", "l", "mn", "r"};
    vector<size_t> result;
    vector<size_t> result2;
    size_t index = 0;

    cin >> word;
    for (string::iterator iter = next(word.begin()); iter < word.end(); ++iter){
        for (size_t i = 0; i < code.size(); ++i){
            for(char c : code[i]){
                if (c == *iter){
                    result.push_back(i + 1);
                }
            }
        }
    }
    while (index < result.size()){
    if (index == 0){
        result2.push_back(result[index]);
        ++index;
        continue;
    }
    if (result[index - 1] != result[index]){
        result2.push_back(result[index]);
    }
    ++index;
    }
    result2.push_back(0);
    result2.push_back(0);
    result2.push_back(0);
    cout << word[0] << result2[0] << result2[1] << result2[2];
    return 0;
}
