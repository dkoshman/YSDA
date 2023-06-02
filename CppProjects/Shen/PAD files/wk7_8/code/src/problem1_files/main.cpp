////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief      Main module for Problem 1: Files.
/// \author     Georgii Zhulikov
/// \author     Sergey Shershakov
/// \version    0.1.0
/// \date       01.02.2021
///             This code is for educational purposes of the course "Introduction
///             to programming" provided by the Faculty of Computer Science
///             at the Higher School of Economics.
///
/// 1) Create a function called sumLines() that obtains an input stream
/// object (given by reference) istream&, reads float numbers from it line by line
/// and sums up numbers from each line. The result is output to a given
/// output stream (given by reference) ostream&.
///
/// Reuse function calcSumFromStream() developed in ex.7 to deal with individual lines!
///
/// In the main program provide two different file stream (both text files) to
/// read from and output to data, correspondingly.
/// http://www.cplusplus.com/doc/tutorial/files/
/// 
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>      // for files

using std::cout;
using std::string;
using std::stringstream;

// TODO: Provide a declaration (a prototype) of the method calcSumFromStream() here.


// TODO: Provide a definition of the method sumLines() here.
double calcSumFromStream(std::istream& inputStream);


// TODO: Provide a definition of the method sumLines() here.
void sumLines(std::istream& in, std::ostream& out)
{
//    bad bit
//    fail bit
//    eof bit
    while(in.good() && !in.eof())
    {
        double lineSum = calcSumFromStream(in);
        out << lineSum << "\n";
    }
}

int main()
{
    using std::cout;
    using std::cin;

    cout << "Workshop 7 Example 3\n\n";
    std::ifstream inputFile("../../data/problem1_files/inp.txt");
        std::ofstream outputFile("../../data/problem1_files/out.txt");


        double x;
        cout << "Workshop 7 Example 1\n\n";

        sumLines(inputFile, outputFile);

        // TODO: Implement the main method here.
        cout << "\n\n";
        inputFile.close();
        outputFile.close();

        return 0;
    }

    // TODO: Implement calcSumFromStream() method here.
    double calcSumFromStream(std::istream& inputStream)
    {
        std::string buffer;
        std::getline(inputStream, buffer);

        std::stringstream ss(buffer);

        double sum = 0;

        while(ss.good() && !ss.eof())
        {
            double a;
            ss >> a;
            sum += a;
        }
        return sum;
    }
