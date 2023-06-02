#ifndef CHAPTER_3
#define CHAPTER_3
#include <notes_utilities.cpp>
#include <string>
#include <sstream>

using namespace std;

void test_throw() {
    throw std::out_of_range("test");
    cout << "hi";
}


void chapter_3() {

        using std::cin;
        string s(5, 'c');
        string s1;
        assert(s == "ccccc" and s1 == "");
        stringstream ss;
        ss << "1st line\n2nd line\n3rd line, end";
        getline(ss, s) >> s1;
        assert(s == "1st line" && s1 == "2nd");
        while (ss >> s);
        assert(s == "end");
        s = "Hello";
        s1 = "Hello World!";
        string s2 = "Hi";
        assert(s < s1 && s1 < s2); // s1 < s2 because 'e' < 'i'
//        s2 = "Hello" + ", " + s2;  // error: + operator can concatenate strings, but not string literals, which
                                     // are of type *const char
        assert(typeid ("hello").name() == (string)"A6_c");
        s2 = s + ' ' + "World"; // ok: return type of first concatenation is string, so second + works
        assert(isalnum('w') && islower('w') && ispunct('\'') && isspace('\v') && isxdigit('A') && isupper(toupper('w')));
        decltype (s1.size()) count = 0;
        for (auto &c : s1)
            count += isupper(c) ? 1 : islower(c = toupper(c));
        assert(count == 2 && s1 == "HELLO WORLD!");
        if (false)
        {
            try {
                test_throw();
            }  catch (exception& e) {
                cout << "caught"; // still gets printed even though an exception is thrown
                throw;
            }
            cout << "test";
            throw;
        }

}
#endif
