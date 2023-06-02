#include <iostream>
#include <fstream>
#include <zoo_sample.h>

int main()
{
    std::ifstream in("in.txt");
    std::cin.rdbuf(in.rdbuf());
    Zoo zoo = createZoo();
    process(zoo);
    return 0;
}
