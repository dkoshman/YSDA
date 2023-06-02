#define MAGIC 100

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <set>

int rand() {  // NOLINT
    throw std::runtime_error("Don't use rand");
}

#include "fixed_set.h"

std::vector<int> ReadSequence() {
    size_t size;
    std::cin >> size;
    std::vector<int> sequence(size);
    for (auto &current: sequence) {
        std::cin >> current;
    }
    return sequence;
}

std::vector<bool> PerformRequests(const std::vector<int> &requests, const FixedSet &set) {
    std::vector<bool> request_answers;
    request_answers.reserve(requests.size());
    for (int request: requests) {
        request_answers.push_back(set.Contains(request));
    }
    return request_answers;
}

void PrintRequestsResponse(const std::vector<bool> &request_answers) {
    for (bool answer: request_answers) {
        std::cout << (answer ? "Yes" : "No") << "\n";
    }
}

void RunTests();

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--testing")) {
        RunTests();
        return 0;
    }

    std::ios::sync_with_stdio(false);

    auto numbers = ReadSequence();
    auto requests = ReadSequence();
    FixedSet set;
    set.Initialize(numbers);
    PrintRequestsResponse(PerformRequests(requests, set));

    return 0;
}

// ========= TESTING ZONE =========

#define ASSERT_EQ(expected, actual) do { \
    auto __expected = expected; \
    auto __actual = actual; \
    if (!(__expected == __actual)) { \
        std::cerr << __FILE__ << ":" << __LINE__ << ": Assertion error" << std::endl; \
        std::cerr << "\texpected: " << __expected << " (= " << #expected << ")" << std::endl; \
        std::cerr << "\tgot: " << __actual << " (= " << #actual << ")" << std::endl; \
        std::cerr << "=========== FAIL ===========\n"; \
        throw std::runtime_error("Check failed"); \
    } \
} while (false)

void Empty() {
    FixedSet set;
    set.Initialize({});
    ASSERT_EQ(false, set.Contains(0));
}

void Simple() {
    FixedSet set;
    set.Initialize({-3, 5, 0, 3, 7, 1});
    ASSERT_EQ(true, set.Contains(0));
    ASSERT_EQ(true, set.Contains(-3));
    ASSERT_EQ(true, set.Contains(1));
    ASSERT_EQ(false, set.Contains(2));
    ASSERT_EQ(false, set.Contains(4));
}

void RepeatInitialize() {
    FixedSet set;
    const int shift = 100;
    int element = 0;
    int last = -1;
    for (int elements_count = 0; elements_count < 10; ++elements_count) {
        std::vector<int> elements;
        for (int i = 0; i < elements_count; ++i) {
            elements.push_back(element++);
        }
        set.Initialize(elements);
        for (auto elem: elements) {
            ASSERT_EQ(true, set.Contains(elem));
        }
        ASSERT_EQ(false, set.Contains(last));
        last = element - 1;
        element += shift;
    }
}

void StressTest() {
    std::mt19937_64 gen(std::random_device{}());
    std::uniform_int_distribution<int32_t> dis(-1'000'000'000, 1'000'000'000);
    auto rand = [&]() { return dis(gen); };
    std::vector<int32_t> vec(100'000);
    generate(begin(vec), end(vec), rand);
    std::set<int32_t> lib_set(vec.begin(), vec.end());
    std::vector<int32_t> vec2(lib_set.begin(),  lib_set.end());
    FixedSet set;
    set.Initialize(vec2);
    for (int32_t i = 0; i < 1'000'000; ++i) {
        auto x = rand();
        ASSERT_EQ(set.Contains(x), lib_set.find(x) != lib_set.end());
    }
}

void Magic() {
#ifdef MAGIC
//    std::cerr << "You've been visited by SymmetricHash Police!\n";
//    std::cerr << "Probably your hash_ table_ is not as good as you think.\n";
//    std::cerr << "No ticket today, but you better be careful.\n\n";
    int first = -1'000'000'000;
    int second = first + MAGIC;
    FixedSet set;
    set.Initialize({first, second});
    ASSERT_EQ(true, set.Contains(first));
    ASSERT_EQ(true, set.Contains(second));
    ASSERT_EQ(false, set.Contains(0));
#endif
}

void RunTests() {
    std::cerr << "Running tests...\n";
    for (int i = 0; i < 1000; ++i) {
        Empty();
        Simple();
        RepeatInitialize();
        Magic();
    }
    std::cerr << "standard Tests are passed!\n";
    StressTest();
    std::cerr << "Tests are passed!\n";
}