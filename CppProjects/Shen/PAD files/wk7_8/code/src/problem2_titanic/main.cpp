////////////////////////////////////////////////////////////////////////////////
// \file
/// \brief      Main module for Problem 2: Titanic.
/// \author     Georgii Zhulikov
/// \author     Sergey Shershakov
/// \version    0.1.0
/// \date       01.02.2021
///             This code is for educational purposes of the course "Introduction
///             to programming" provided by the Faculty of Computer Science
///             at the Higher School of Economics.
///
/// 1) Define an alias VecStrings for the std::vector<std::string> datatype
/// using the typedef keyword.
///
/// 2) Create a function called toCountSurvived that obtains an input stream
/// object (given by reference) istream& (input.csv), reads the whole file
/// line by line and saves surnames ("Braund; Mr. Owen Harris" will be just
/// "Braund") of survived people from input.csv (Survived column).
/// The function returns data of type VecStrings -- vector of surnames of survivors.
///
/// Use intermediate functions in task 2 to do the following:
/// 2.1) Extract data (surname and whether the person survived or not) from one line of input.
/// 2.2) Extract surname from a string containing full name.
///
///
/// 3) Create a function printVec.
/// The function prints the content of the given vector out to the standard output.
/// It should takes a vector as argument by reference and print the value of the
/// elements as well as their enumeration.
/// 1) Name_1
/// 2) Name_2
/// ...
/// N) Name_n
///
///
///
/// 4) Create a function called getFareForClass that takes an input stream object
/// istream& and an integer number representing class (Pclass, 1 to 3), reads the stream
/// until the end and returns the mean fare value of people of the given class.
/// The function returns a single value of type double -- the mean fair value.
/// Use at least two intermediate functions in problem 4.
///
///
/// Additional problems
///
/// 5) Create a function called genThreeDigitNumber(const int& randomState).
/// The function returns a random three digit number as std::string.
/// Use:
/// std::mt19937 gen(randomState);
/// std::uniform_int_distribution<int> distr(0,9);
/// int rNum = distr(gen); // random number
///
/// 6) Create a new vector newVec as VecStrings and fill it with random numbers.
/// newVec size should be the same as the size of the vector obtained from toCountSurvived
///
///
///   Workshop 8
///
/// 6) Reverse the vector containing names of surviving passengers using std::reverse.
/// Sort this vector using std::sort
/// Include library <algorithm> to access these functions.
///
/// 7) Implement a function printVecIter that takes two vector iterators as arguments
/// and prints the elements of the underlying vector. The iterators should represent
/// the start and the end of the vector.
///
/// 8) Use a regular iterator and a reverse iterator (.rbegin()) to print the vector
/// containing survivor names in a straightforward order and in a reverse order
/// with the function printVecIter.
///
/// 9) Using the sorted list of surnames find the first and last surname that starts with
/// the letter "H".
/// Create a new vector and use functions std::find and std::copy to copy all surnames
/// starting with "H" into it.
///
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>


typedef std::vector<std::string> VecStrings;

void printVec(const VecStrings& vecStrings)
{
    int i = 1;
    for(const std::string& str : vecStrings)
    {
        std::cout << i << ") " << str << std::endl;
        ++i;
    }
}

void printVecIter(VecStrings::iterator start, VecStrings::iterator end){
    int i = 0;
    while(start != end){
        std::cout << ++i << ") " << *start << std::endl;
        ++start;
    }
}

void printVecIter(VecStrings::reverse_iterator start, VecStrings::reverse_iterator end){
    int i = 0;
    while(start != end){
        std::cout << ++i << ") " << *start << std::endl;
        ++start;
    }
}

std::string convertName(std::string fullname)
{
    std::string surname;
    std::stringstream ss(fullname);
    std::getline(ss, surname, ';');
    return surname;
}

void extractData(std::istream& in, std::string& surname, bool& survived)
{
    std::string buffer;

    std::getline(in, buffer, ','); // passengerID
    std::getline(in, buffer, ','); // survived

    if (buffer == "1")
    {
        survived = true;
    }
    else
    {
        survived = false;
    }

    std::getline(in, buffer, ','); // pclass
    std::getline(in, buffer, ','); // full name
    surname = convertName(buffer);
}


VecStrings toCountSurvived(std::istream& in)
{
    VecStrings survivorNames;
    std::string buffer;
    while(in.good())
    {
        std::getline(in, buffer);
        if (!in.good())
        {
            break;
        }
        std::stringstream lineStream(buffer);
        bool survived;
        std::string surname;
        extractData(lineStream, surname, survived);
        if (survived)
        {
            survivorNames.push_back(surname);
        }
    }
    return survivorNames;
}

void findH(VecStrings::iterator start, VecStrings::iterator end, std::vector<int>& indices){
    int i = 0;
    int j = 0;
    std::string name = "";
    while(start != end){
        name = *start;
        if (name.at(0) == 'H'){
            if (j == 0){
                indices.push_back(i);
            }
            ++j;
        }
        ++i;
        ++start;
    }
    indices.push_back(indices[0] + j);
}

int main ()
{
    // Change "/" in path to "\\" if you are using Windows
    const std::string INP_FILE_NAME = "../../data/problem2_titanic/titanic.csv";
    std::ifstream inputFile;
    inputFile.open(INP_FILE_NAME);

    VecStrings vecNames = toCountSurvived(inputFile);
    VecStrings newNames;
    std::sort(vecNames.begin(), vecNames.end());
    std::vector<int> indices;
    findH(vecNames.begin(), vecNames.end(), indices);
    std::copy(vecNames.begin() + indices[0], vecNames.begin() + indices[1], std::back_inserter(newNames));
    printVecIter(newNames.begin(), newNames.end());
    //findH(newNames.begin(), newNames.end());
    //printVecIter(vecNames.begin(), vecNames.end());
    //printVecIter(vecNames.begin(), vecNames.end());
    //printVecIter(vecNames.rbegin(), vecNames.rend());
    //std::reverse(vecNames.begin(), vecNames.end());
    //printVecIter(vecNames.begin(), vecNames.end());

}
