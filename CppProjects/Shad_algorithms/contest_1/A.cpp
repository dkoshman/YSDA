// https://contest.yandex.ru/contest/29039/problems/

#include <assert.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using InputType = std::vector<int32_t>;
using OutputType = std::vector<int32_t>;

void SetUp() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

InputType Input(std::istream &in = std::cin) {
    int n_elements = 0;
    in >> n_elements;
    InputType sequence(n_elements);
    for (auto &element : sequence) {
        in >> element;
    }
    return sequence;
}

void Output(OutputType answer, std::ostream &out = std::cout) {
    for (const auto &item : answer) {
        out << item << ' ';
    }
    out << std::endl;
}

bool inline IsAlternatingTriple(int32_t first, int32_t second, int32_t third) {
    return (first < second and second > third) or (first > second and second < third);
}

bool inline IsAlternatingContinuation(const OutputType &answer, int32_t next) {
    if (answer.empty()) {
        return true;
    }
    if (answer.size() == 1) {
        return answer.back() != next;
    }
    return IsAlternatingTriple(answer.end()[-2], answer.end()[-1], next);
}

bool inline IsAlternatingContinuation(const OutputType &answer, int32_t next,
                                      int32_t one_after_next) {
    if (answer.empty()) {
        return next != one_after_next;
    }
    return IsAlternatingContinuation(answer, next) and
           IsAlternatingTriple(answer.back(), next, one_after_next);
}

OutputType FindAllPeaksAndBoundaries(const InputType &input) {
    OutputType peaks;
    for (auto element : input) {
        if (IsAlternatingContinuation(peaks, element)) {
            peaks.push_back(element);
        } else {
            peaks.back() = element;
        }
    }
    return peaks;
}

OutputType FindAlternatingSequenceWithMinimalIndices(const InputType &input,
                                                     const OutputType &peaks) {
    OutputType answer{input[0]};
    size_t current_peak_id = 2;
    // Invariant: @answers is the optimal alternating sequence up to @element
    // that can be extended to maximum possible length.
    for (auto element : input) {
        if (current_peak_id >= peaks.size()) {
            if (IsAlternatingContinuation(answer, element)) {
                answer.push_back(element);
            }
        } else if (IsAlternatingContinuation(answer, element, peaks[current_peak_id])) {
            answer.push_back(element);
            ++current_peak_id;
        }
    }
    return answer;
}

OutputType Solve(const InputType &input) {
    auto peaks = FindAllPeaksAndBoundaries(input);
    return FindAlternatingSequenceWithMinimalIndices(input, peaks);
}

std::string StripWhitespace(const std::string &line) {
    const char *white_space = " \t\v\r\n";
    std::size_t start = line.find_first_not_of(white_space);
    std::size_t end = line.find_last_not_of(white_space);
    return start > end ? std::string() : line.substr(start, end - start + 1);
}

void Check(const std::string &test_case, const std::string &expected) {
    std::stringstream input_stream(test_case);
    std::stringstream output_stream;
    Output(Solve(Input(input_stream)), output_stream);
    auto output = StripWhitespace(output_stream.str());
    auto expected_trimmed = StripWhitespace(expected);
    if (output != expected_trimmed) {
        std::cerr << "Check failed, expected:\n\"";
        std::cerr << expected_trimmed << "\"\nReceived:\n\"";
        std::cerr << output << "\"\n";
        assert(false);
    }
}

void Test() {
    Check("10 1 4 2 3 5 8 6 7 9 10", "1 4 2 8 6 7");
    Check("5 1 2 3 4 5", "1 2");
    Check("9 1 3 5 4 6 4 5 3 1", "1 5 4 6 4 5 3");
    Check("3 4 2 1", "4 2");
    Check("10 1 3 5 -2 1 -1 2 1 2 3", "1 3 -2 1 -1 2 1 2");
    Check("3 2 2 2", "2");
    Check("3 2 -2 2", "2 -2 2");
    Check("1 100", "100");
    Check("10 1 17 5 10 13 15 10 5 16 8", "1 17 5 10 5 16 8");
    std::cout << "OK\n";
}

int main(int argc, char *argv[]) {
    SetUp();
    if (argc > 1 and std::strcmp(argv[1], "test") == 0) {
        Test();
    } else {
        Output(Solve(Input()));
    }
    return 0;
}
