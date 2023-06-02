#ifndef CHAPTER_2
#define CHAPTER_2
#include <notes_utilities.cpp>
#include <vector>
#include <random>

using namespace std;

constexpr double cexpr_f(int i){
    return 3.14 * i;
}

void chapter_2() {
    {
        int a = -1;
        unsigned sb = 1;
        print(a * sb); // converted to unsigned even though int is first
        int i = 42;
        const int &r1 = i;
        const int &r2 = 42; // const references to temporary objects are allowed - so they are no longer temporary
        cout << r2 << ',' << &r2 << endl;
        cout << r1 << ',' << &r1 << endl;
        i = 7;
        cout << r1 << ',' << &r1 << endl;
        const int &r3 = r1 * 2;
        cout << r3 << ',' << &r3 << endl;
        double dval = 3.14;
        const int &ri = dval; //
        double cd = 3.14;
        const double *cd_p = &cd;
        cd /= 2;
        print(*cd_p);
        using triple = double;
        constexpr triple cexpr = cexpr_f(7);
        typedef int basket_of_apples;
        constexpr basket_of_apples i_cx = cexpr;
        bitset<i_cx> bt(i);
        print(bt);
        print(sizeof(bt));
        typedef char *pstring; // base type of pstring is pointer
        const pstring cstr = 0; // cstr is a constant pointer to char
        // and not pointer to const char, as it would have been
        // if we replace pstring with its literal representation:
        // const (char *)cstr -- wrong
        // Because base type of pstring is pointer, and const here -
        // const pstring cstr = 0;
        // is top level, so it modifies the base type
        const pstring *ps; // ps is a pointer to a constant pointer to char
        auto i1 = 3.14, i2 = 0.0; // even with auto all variables in a declaration have the same type
        assert(0. == .0 && 0 == 00000.00000);
        const int ci = i;
        auto b = ci; // b is an int (top-level const in ci is dropped)
        // Auto drops top-level const -- I guess cause it can initialize non-const
        // from const and non-const is more generic. But it keeps low-level const upon creating
        // pointers and references otherwise it would allow const objects to be changed.
        auto e = &ci; // e is const int*(& of a const object is low-level const)
        const auto c = i; // explicit top-level const
    }
    {
        const int i = 42;
        auto j = i; const auto &k = i; auto *p = &i; const auto j2 = i, &k2 = i;
        print(typeid(p).name());
        decltype(print(0)) *v;
        print(typeid(v).name()); // pointer to void
        decltype(cexpr_f(0)) dd = 0;
        decltype(&i) ii = 0; // ii is const int*
        print(typeid(ii).name());
        decltype(*p) iip = i; // iip is a const int&
        int a = 0;
        (a) = 1; // (a) is an assignable expression, ie an lvalue
        decltype((a)) ap = a; // so ap is a reference, int&
        decltype(++a) ap2 = a; // same: ++a is lvalue, ap2 is int&
        decltype(a++) ap3 = 42; // but here a++ is not an lvalue, so ap3 is int
        decltype(a = a) ap4 = a; // lvalue
    }
    {
        struct Str {int a; int b = 7;} S_exemplar; // Thats why we need semicolon after class definiton--
        // because we can immediately declare variables of that class. But it's bad practice.
        S_exemplar.a = 42;
        print(S_exemplar.b);
        print(sizeof(S_exemplar)); // size in bytes
        print(string("a") <= string("b"));
        vector<int> v = {1, 2, 3};
        print(v.capacity()); // 3
        assert(v.size() == v.capacity());
        v.push_back(4);
        print(v.capacity()); // 6 = 3 * 2
        v.reserve(42);
        assert(v.capacity() == 42);
    }
    {
        assert(-1 >> 1 == -1 && -123 >> 31 == -1); // Undefined behavior -- bitshift doesn't affect the
                                   // first bit of negative number, which is stored as two's complement:
                                   // -1 = 1111 1111, -2 = 1111 1110, -x = -(~x + 1)
                                   // This can be used as a hack:
        int x = 100;
        x += ~((x - 128) >> 31) & x; // doubles x iff x >= 128 -- faster than if branch
        assert(x == 100);
        x = 200;
        x += ~((x - 128) >> 31) & x;
        assert(x == 400);
    }
    {
        // Generate data
        const unsigned arraySize = 32768;
        int data[arraySize];
        std::mt19937 mt(time(nullptr));

        for (unsigned c = 0; c < arraySize; ++c)
            data[c] = mt() % 256;



        // Test
        clock_t start = clock();
        long long sum = 0;
        for (unsigned i = 0; i < 1000; ++i)
        {
            for (unsigned c = 0; c < arraySize; ++c)
            {   // Primary loop
                if (data[c] >= 128)
                    sum += data[c];
            }
        }

        double elapsedTime = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
        print("Processing if branches with random data:");
        print(elapsedTime);

        // !!! With this, the next loop runs faster.
        std::sort(data, data + arraySize);
        start = clock();
        sum = 0;

        for (unsigned i = 0; i < 1000; ++i)
        {
            for (unsigned c = 0; c < arraySize; ++c)
            {   // Primary loop
                if (data[c] >= 128)
                    sum += data[c];
            }
        }

        elapsedTime = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
        print("Sorted data allows for efficient branch predictions:");
        print(elapsedTime);

    }
}
#endif
