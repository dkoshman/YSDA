#include <iostream>
#include "divmod.cpp"
#include "euclidian.cpp"
#include "besu.cpp"
#include "chinese.cpp"
#include "fibbonachi.cpp"
#include "prime_decomposition.cpp"
#include "shen_1_1_29.cpp"
#include "shen_1_1_31.cpp"
#include "shen_1_1_32.cpp"
#include "shen_1_1_33.cpp"
#include "shen_1_1_35.cpp"
#include "shen_1_2_13.cpp"
#include "shen_1_2_22.cpp"
#include "shen_1_2_27.cpp"
#include "shen_1_2_35.cpp"
#include "shen_1_2_30.cpp"
#include "shen_1_2_32.cpp"
#include "shen_1_3_4.cpp"
#include "shen_2_1_1.cpp"
#include "shen_2_2_1.cpp"
#include "shen_2_3_1.cpp"
#include "shen_2_4_1.cpp"
#include "shen_2_5_1.cpp"
#include "shen_2_5_2.cpp"
#include "shen_2_6_1.cpp"
#include "shen_2_6_2.cpp"
#include "shen_3_2_1.cpp"
#include "shen_4_2_1.cpp"
//#include ".cpp"
//#include ".cpp"

using namespace std;

typedef int (*Function) ();

class Program {
public:
    string description;
    Function execute;
    Program (string, Function);
};

Program::Program(string s, Function f){
    description = s;
    execute = f;
};

int main()
{
    Program programs[] = {
        Program("divmod", divmod),
        Program("euclidian algorithm", euclidian),
        Program("besu algorithm", besu),
        Program("chinese remainder theorem", chinese),
        Program("fibbonachi -- log complexity", fibbonachi),
        Program("prime decomposition", prime_decomposition),
        Program("shen_1_1_29: count number of natural solutions to x^2 + y^2 < n", shen_1_1_29),
        Program("shen_1_1_31: find period of 1/n fraction", shen_1_1_31),
        Program("shen_1_1_32: find period of sequence", shen_1_1_32),
        Program("shen_1_1_33: calculate value of recusive function", shen_1_1_33),
        Program("shen_1_1_35: calculate a mod b using only multiplication and division by 2"
                , shen_1_1_35),
        Program("shen_1_2_13: compute value of polynomial of degree n and its derivative at point x", shen_1_2_13),
        Program("shen_1_2_22: given n and two nondecreasing arrays x and y, find closest sum x[i] + y[j] to n", shen_1_2_22),
        Program("shen_1_2_27: binary search", shen_1_2_27),
        Program("shen_1_2_30: determine the parity of permutation, find inverse", shen_1_2_30),
        Program("shen_1_2_32: partially sort array in three parts: less than x, equal x and more than x", shen_1_2_32),
        Program("shen_1_2_35: find max in all m consecutive elements of array", shen_1_2_35),
        Program("shen_1_3_4: find longest increasing subarray", shen_1_3_4),
        Program("shen_2_1_1: print all arrays of size k consisting of numbers 1..n", shen_2_1_1),
        Program("shen_2_2_1: print all permutations of size n", shen_2_2_1),
        Program("shen_2_3_1: print all k element subsets of 1..n", shen_2_3_1),
        Program("shen_2_4_1: print all decompositions of n into positive terms", shen_2_4_1),
        Program("shen_2_5_1: print all arrays of size k consisting of numbers 1..n in a chain "
                "of smallest possible changes", shen_2_5_1),
        Program("shen_2_5_2: print all permutations of 1..n in a chain "
        "of smallest possible changes", shen_2_5_2),
        Program("shen_2_6_1: print all sequences corresponding to Catalan numbers", shen_2_6_1),
        Program("shen_2_6_2: print all ways to sequence multiplication between n numbers standing in fixed order", shen_2_6_2),
        Program("shen_3_2_1: use tree traversal to determine if s is a sum of elements of array 1..n", shen_3_2_1),
        Program("shen_4_2_1: sort an array using tree sort", shen_4_2_1),
//Program("", ),
//Program("", ),
    };
    int pro_n = 0;
    string inp = "";

    for (Program p : programs){
        cout << pro_n++ << " " << p.description << '\n';
    }
    cout << "Which one?\n";

#ifdef QT_DEBUG
    inp = "21";
#else
    cin >> inp;
#endif

    do {
        try {pro_n = stoi(inp);}
        catch (const std::exception& e){}
        programs[pro_n].execute();
        cout << "Again? q to exit\n";
        cin >> inp;
    } while (inp != "q");

    return 0;
}
