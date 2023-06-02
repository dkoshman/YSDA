#include <iostream>
#include <vector>
#include <cctype>

using namespace std;

int main()
{
    std::string password;
    bool valid = true;
    std::vector<bool> categories = {0, 0, 0, 0};
    int cat_sum = 0;

    cin >> password;
    if (8 <= password.length() && password.length() <= 14){
        for (char c : password){
            if (c < 33 || c > 127){
                valid = false;
                break;
            }
            if (std::islower(c)){
                categories[0] = 1;
            } else if (std::isupper(c)){
                categories[1] = 1;
            } else if (std::isdigit(c)){
                categories[2] = 1;
            } else {
                categories[3] = 1;
            }
        }
    } else {
        valid = false;
    }
    for (bool i : categories){
        cat_sum += i;
    }
    if (cat_sum < 3){
        valid = false;
    }
    cout << (valid ? "YES" : "NO");
    return 0;
}
