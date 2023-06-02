#include "notes_utilities.cpp"
#include <fstream>
#include <iostream>


using namespace std;
int global;
int reused = 42;
extern const int f = 7;
int d = 9;

void chapter_1()
{
    std::ifstream in("in.txt");
    std::ofstream out("out.txt");
    std::streambuf *cinbuf = std::cin.rdbuf(); //save old buf
    std::cin.rdbuf(in.rdbuf());


    cerr << "Error" << endl;
    clog << "Log" << endl;
    assert(/* "/*" "*/" /* "/*" */ == " /* ");
    int a = 0;
    class F{
    public:
        int &a;
        F(int &a) : a{a}{};
        bool operator() (){return ++a < 5;};
    };
    F for_condition(a);
    for (cout << "start" << endl; for_condition(); cout << "step,")
        {;;;;};;;;
    a = 5;
    cout << endl << "a=" << a;  // outputs "a=5"
    cin >> a;                   // input "a"
    cout << "a=" << a << endl;  // outputs "a=0"
    std::string s = "test";
    std::string *s_ptr = &s;
    cout << s_ptr << endl;
    cout << (a += 3.14) << endl;
    unsigned b = 1;
    a = -010;
    string str = u8"ф";
    assert(str == "ф");
    string str1 = "a";
    assert(str1 == u8"a");
    str1 = "\141\40" "separated\40"

           "string literal?\'\n\r\t\v\a\"";
    cout << str1 << endl;
    assert(16 == 020 && 020 == 0x10); // decimal, octal and hexadecimal
    assert(3.14L == 3.14e00000000000000000000000000000000000000000L);
    cout << a << ", " << a * b << endl;
    long double ld = 3.1415;
//  error: when using initializer lists, loss of data is not permited by the compiler
//      int c = {ld};
    int c = {3};
    int _;
    cout << "uninitialized: " << _ << ", global variable: " << global << endl;
    extern int d;
//        d = -1;
    print(d);
    int reused = 0;
    cout << ::reused << ',' << reused << endl;
    int i = 10, sum = 0;
    for (int i = 0; i != 10; ++i)
        sum += 1;
    std::cout << i << " " << sum << std::endl;
    int e = 1;
    int &ref_e = e;
    cout << &ref_e << ',' << &e << endl; // same adress
    int *ip, *&r = ip; // read from right to left: r is a reference to a pointer to an int
    ip = &e;
    cout << *r << endl;
    e = 2;
    r = &e;
    print(*ip); // ip changed to 2 because r is a reference to ip
    extern const int f;
    extern const int f;
    cout << f << endl;
    in.close();
    out.close();
}
