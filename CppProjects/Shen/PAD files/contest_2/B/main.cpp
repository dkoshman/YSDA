#include <iostream>
#include <algorithm>

using namespace std;

int main()
{
    std::string line;
    std::string line_reversed;

    std::getline(std::cin, line);
    std::string::iterator end_pos = std::remove(line.begin(), line.end(), ' ');
    line.erase(end_pos, line.end());
    line_reversed.resize(line.size());
    std::reverse_copy(line.begin(), line.end(), line_reversed.begin());
    cout << ((line == line_reversed) ? "yes" : "no");
    return 0;
}
