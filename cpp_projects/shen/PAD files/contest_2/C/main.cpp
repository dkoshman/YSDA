#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>

using namespace std;

int main()
{
//    "C:\\Users\\Surface\\SkyDrive\\Documents\\qt\\contest_2\\C\\input.txt";
    const std::string INP_FILE_NAME = "input.txt";
    std::string line;
    int lines = 0;
    int words = 0;
    int letters = 0;
    bool prev_alpha = false;
    bool curr_alpha = false;

    std::ifstream inputFile;
    inputFile.open(INP_FILE_NAME);
    while (std::getline(inputFile, line)){
        ++lines;
        for (char c : line){
            curr_alpha = false;
            if (isalpha(c)){
                curr_alpha = true;
                ++letters;
            }
            if (!prev_alpha && curr_alpha){
                ++words;
            }
            prev_alpha = curr_alpha;
        }
        prev_alpha = false;
    }
    inputFile.close();
    cout << "Input file contains:"  << endl
         << letters << " letters"   << endl
         << words   << " words"     << endl
         << lines   << " lines"     << endl;
    return 0;
}
