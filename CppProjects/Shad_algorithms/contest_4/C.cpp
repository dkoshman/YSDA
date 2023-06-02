#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rng {

uint32_t GetSeed() {
    auto random_device = std::random_device{};
    static auto seed = random_device();
    return seed;
}

void PrintSeed(std::ostream &ostream = std::cerr) {
    ostream << "Seed = " << GetSeed() << std::endl;
}

std::mt19937 *GetEngine() {
    static std::mt19937 engine(GetSeed());
    return &engine;
}

}  // namespace rng

namespace io {

class InputType {
public:
    std::vector<int32_t> numbers_in_preorder;

    InputType() = default;

    explicit InputType(std::istream &in) {
        int32_t count_elements = 0;
        in >> count_elements;
        numbers_in_preorder.resize(count_elements);
        for (auto &number : numbers_in_preorder) {
            in >> number;
        }
    }
};

class OutputType {
public:
    std::vector<int32_t> numbers_in_inorder;
    std::vector<int32_t> numbers_in_postorder;

    OutputType() = default;

    explicit OutputType(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            numbers_in_postorder.push_back(item);
        }

        auto half_size = static_cast<int32_t>(numbers_in_postorder.size() / 2);
        numbers_in_inorder = std::vector<int32_t>{numbers_in_postorder.begin() + half_size,
                                                  numbers_in_postorder.end()};
        numbers_in_postorder = std::vector<int32_t>{numbers_in_postorder.begin(),
                                                    numbers_in_postorder.begin() + half_size};
    }

    OutputType(std::vector<int32_t> numbers_in_inorder, std::vector<int32_t> numbers_in_postorder)
        : numbers_in_inorder{std::move(numbers_in_inorder)},
          numbers_in_postorder{std::move(numbers_in_postorder)} {
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : numbers_in_postorder) {
            out << item << ' ';
        }
        out << '\n';
        for (auto item : numbers_in_inorder) {
            out << item << ' ';
        }
        return out;
    }

    bool operator!=(const OutputType &other) const {
        return std::tie(numbers_in_postorder, numbers_in_inorder) !=
               std::tie(other.numbers_in_postorder, numbers_in_inorder);
    }
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

}  // namespace io

using io::InputType, io::OutputType;

struct CorrespondingOrderIntervals {
    int32_t size = 0;
    int32_t preorder_begin = 0;
    int32_t inorder_begin = 0;
    int32_t postorder_begin = 0;

    explicit CorrespondingOrderIntervals(int32_t size) : size{size} {
    }

    CorrespondingOrderIntervals(int32_t size, int32_t preorder_begin, int32_t inorder_begin,
                                int32_t postorder_begin)
        : size{size},
          preorder_begin{preorder_begin},
          inorder_begin{inorder_begin},
          postorder_begin{postorder_begin} {
    }

    [[nodiscard]] int32_t GetPreorderEnd() const {
        return preorder_begin + size;
    }

    [[nodiscard]] int32_t GetInorderEnd() const {
        return inorder_begin + size;
    }

    [[nodiscard]] int32_t GetPostorderEnd() const {
        return postorder_begin + size;
    }

    [[nodiscard]] int32_t GetPreorderRootPosition() const {
        return preorder_begin;
    }

    [[nodiscard]] int32_t GetInorderLeftSubtreeBegin() const {
        return inorder_begin;
    }

    [[nodiscard]] int32_t GetPostorderLeftSubtreeBegin() const {
        return postorder_begin;
    }
};

struct CorrespondingOrderIntervalsWithChildren : public CorrespondingOrderIntervals {
    int32_t right_subtree_preorder_begin = 0;

    CorrespondingOrderIntervalsWithChildren(CorrespondingOrderIntervals intervals,
                                            int32_t right_subtree_preorder_begin)
        : CorrespondingOrderIntervals{intervals},
          right_subtree_preorder_begin{right_subtree_preorder_begin} {
    }

    [[nodiscard]] int32_t GetRightSubtreeSize() const {
        return GetPreorderEnd() - right_subtree_preorder_begin;
    }

    [[nodiscard]] int32_t GetLeftSubtreeSize() const {
        return size - 1 - GetRightSubtreeSize();
    }

    [[nodiscard]] int32_t GetPreorderLeftSubtreeBegin() const {
        return GetPreorderRootPosition() + 1;
    }

    [[nodiscard]] int32_t GetPreorderRightSubtreeBegin() const {
        return GetPreorderLeftSubtreeBegin() + GetLeftSubtreeSize();
    }

    [[nodiscard]] int32_t GetInorderRootPosition() const {
        return GetInorderLeftSubtreeBegin() + GetLeftSubtreeSize();
    }

    [[nodiscard]] int32_t GetInorderRightSubtreeBegin() const {
        return GetInorderRootPosition() + 1;
    }

    [[nodiscard]] int32_t GetPostorderRightSubtreeBegin() const {
        return GetPostorderLeftSubtreeBegin() + GetLeftSubtreeSize();
    }

    [[nodiscard]] int32_t GetPostorderRootPosition() const {
        return GetPostorderRightSubtreeBegin() + GetRightSubtreeSize();
    }

    [[nodiscard]] std::pair<CorrespondingOrderIntervals, CorrespondingOrderIntervals>
    SplitIntoCorrespondChildIntervals() const {

        CorrespondingOrderIntervals left{GetLeftSubtreeSize(), GetPreorderLeftSubtreeBegin(),
                                         GetInorderLeftSubtreeBegin(),
                                         GetPostorderLeftSubtreeBegin()};

        CorrespondingOrderIntervals right{GetRightSubtreeSize(), GetPreorderRightSubtreeBegin(),
                                          GetInorderRightSubtreeBegin(),
                                          GetPostorderRightSubtreeBegin()};

        return {left, right};
    }
};

class TreeTraversalOrders {
public:
    std::vector<int32_t> numbers_in_preorder;
    std::vector<int32_t> numbers_in_inorder;
    std::vector<int32_t> numbers_in_postorder;

    void ConvertFromPreorder(std::vector<int32_t> new_numbers_in_preorder) {
        numbers_in_preorder = std::move(new_numbers_in_preorder);

        for (auto other_order : {&numbers_in_inorder, &numbers_in_postorder}) {
            other_order->clear();
            other_order->resize(numbers_in_preorder.size());
        }

        auto intervals =
            CorrespondingOrderIntervals{static_cast<int32_t>(numbers_in_preorder.size())};
        ConvertCorrespondingIntervalsFromPreorder(intervals);
    }

private:
    [[nodiscard]] int32_t FindBeginOfRightSubtreeIntervalFromPreorder(
        CorrespondingOrderIntervals intervals) const {

        auto begin_search = intervals.preorder_begin + 1;
        auto end_search = intervals.GetPreorderEnd();
        auto root = numbers_in_preorder[intervals.GetPreorderRootPosition()];

        while (begin_search != end_search) {
            auto mid = (begin_search + end_search) / 2;
            if (numbers_in_preorder[mid] < root) {
                begin_search = mid + 1;
            } else {
                end_search = mid;
            }
        }

        return begin_search;
    }

    void ConvertCorrespondingIntervalsFromPreorder(CorrespondingOrderIntervals intervals) {
        while (intervals.size > 0) {
            auto right_subtree_begin = FindBeginOfRightSubtreeIntervalFromPreorder(intervals);

            CorrespondingOrderIntervalsWithChildren intervals_with_children{intervals,
                                                                            right_subtree_begin};

            auto root = numbers_in_preorder[intervals_with_children.GetPreorderRootPosition()];
            numbers_in_inorder[intervals_with_children.GetInorderRootPosition()] = root;
            numbers_in_postorder[intervals_with_children.GetPostorderRootPosition()] = root;

            auto [left_interval, right_interval] =
                intervals_with_children.SplitIntoCorrespondChildIntervals();

            if (left_interval.size < right_interval.size) {
                intervals = right_interval;
                ConvertCorrespondingIntervalsFromPreorder(left_interval);
            } else {
                intervals = left_interval,
                ConvertCorrespondingIntervalsFromPreorder(right_interval);
            }
        }
    }
};

OutputType Solve(const InputType &input) {
    OutputType output;

    TreeTraversalOrders tree_traversal_orders{};
    tree_traversal_orders.ConvertFromPreorder(input.numbers_in_preorder);

    return OutputType{tree_traversal_orders.numbers_in_inorder,
                      tree_traversal_orders.numbers_in_postorder};
}

namespace test {

class WrongAnswerException : public std::exception {
public:
    WrongAnswerException() = default;

    explicit WrongAnswerException(std::string const &message) : message{message.data()} {
    }

    [[nodiscard]] const char *what() const noexcept override {
        return message;
    }

    const char *message{};
};

class NotImplementedError : public std::logic_error {
public:
    NotImplementedError() : std::logic_error("Function not yet implemented."){};
};

OutputType BruteForceSolve(const InputType &input) {
    throw NotImplementedError{};
}

struct TestIo {
    InputType input;
    std::optional<OutputType> optional_expected_output = std::nullopt;

    explicit TestIo(InputType input) {
        try {
            optional_expected_output = BruteForceSolve(input);
        } catch (const NotImplementedError &e) {
        }
        this->input = std::move(input);
    }

    TestIo(InputType input, OutputType output)
        : input{std::move(input)}, optional_expected_output{std::move(output)} {
    }
};

void GenerateRandomPreorder(std::vector<int32_t> *numbers, int32_t begin, int32_t end, int32_t min,
                            int32_t max) {
    if (end == begin) {
        return;
    }
    if (min == max) {
        for (int32_t i = begin; i < end; ++i) {
            (*numbers)[i] = min;
        }
    } else {
        auto root_distribution = std::uniform_int_distribution<>{min, max};
        auto root = root_distribution(*rng::GetEngine());
        (*numbers)[begin] = root;

        auto size_distribution = std::uniform_int_distribution<>{0, end - begin - 1};
        auto left_size = root == min ? 0 : size_distribution(*rng::GetEngine());

        GenerateRandomPreorder(numbers, begin + 1, begin + 1 + left_size, min, root - 1);
        GenerateRandomPreorder(numbers, begin + 1 + left_size, end, root, max);
    }
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_numbers = test_case_id * 10;
    int32_t max_number_value = test_case_id * 10;

    InputType input;
    input.numbers_in_preorder.resize(n_numbers);
    GenerateRandomPreorder(&input.numbers_in_preorder, 0, n_numbers, 0, max_number_value);

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    int32_t n_numbers = 100'000;
    int32_t max_number_value = 1'000'000'000;

    InputType input;
    input.numbers_in_preorder.resize(n_numbers);
    GenerateRandomPreorder(&input.numbers_in_preorder, 0, n_numbers, 0, max_number_value);

    return TestIo{input};
}

class TimeItInMilliseconds {
public:
    std::chrono::time_point<std::chrono::steady_clock> begin;
    std::chrono::time_point<std::chrono::steady_clock> end;

    TimeItInMilliseconds() {
        Begin();
    }

    void Begin() {
        begin = std::chrono::steady_clock::now();
    }

    int64_t End() {
        end = std::chrono::steady_clock::now();
        return Duration();
    }

    [[nodiscard]] int64_t Duration() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }
};

int64_t Check(const TestIo &test_io) {

    TimeItInMilliseconds time;
    auto output = Solve(test_io.input);
    time.End();

    auto sorted_numbers = test_io.input.numbers_in_preorder;
    std::sort(sorted_numbers.begin(), sorted_numbers.end());
    if (output.numbers_in_inorder != sorted_numbers) {
        Solve(test_io.input);
        throw WrongAnswerException{};
    }

    if (test_io.optional_expected_output) {
        auto &expected_output = test_io.optional_expected_output.value();

        if (output != expected_output) {
            Solve(test_io.input);
            std::stringstream ss;
            ss << "\n==================================Expected==================================\n"
               << expected_output
               << "\n==================================Received==================================\n"
               << output << "\n";
            throw WrongAnswerException{ss.str()};
        }
    }

    return time.Duration();
}

int64_t Check(const std::string &test_case, const std::string &expected) {
    std::stringstream input_stream{test_case};
    return Check(TestIo{InputType{input_stream}, OutputType{expected}});
}

struct Stats {
    double mean = 0;
    double std = 0;
    double max = 0;
};

template <class Iterator>
Stats ComputeStats(Iterator begin, Iterator end) {
    auto size = end - begin;
    if (size == 0) {
        throw std::invalid_argument{"Empty container."};
    }

    auto mean = std::accumulate(begin, end, 0.0) / size;

    double std = 0;
    for (auto i = begin; i != end; ++i) {
        std += (*i - mean) * (*i - mean);
    }
    std = std::sqrt(std / size);

    auto max = static_cast<double>(*std::max_element(begin, end));

    return Stats{mean, std, max};
}

void Test() {
    rng::PrintSeed();

    Check(
        "7\n"
        "4 2 1 3 6 5 7\n",
        "1 3 2 5 7 6 4\n"
        "1 2 3 4 5 6 7");
    Check(
        "6\n"
        "5 3 2 3 5 6\n",
        "2 3 3 6 5 5\n"
        "2 3 3 5 5 6");
    Check(
        "1\n"
        "6\n",
        "6\n6");
    Check(
        "0\n"
        "\n",
        "\n");

    std::cerr << "Basic tests OK\n";

    std::vector<int64_t> durations;
    TimeItInMilliseconds time_it;

    int32_t n_random_test_cases = 100;
    for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateRandomTestIo(test_case_id)));
    }

    std::cerr << "Random tests OK\n";

    int32_t n_stress_test_cases = 10;
    for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateStressTestIo(test_case_id)));
    }

    std::cerr << "Stress tests tests OK\n";

    auto duration_stats = ComputeStats(durations.begin(), durations.end());
    std::cerr << "Solve duration stats in milliseconds:\n"
              << "\tMean:\t" + std::to_string(duration_stats.mean) << '\n'
              << "\tStd:\t" + std::to_string(duration_stats.std) << '\n'
              << "\tMax:\t" + std::to_string(duration_stats.max) << '\n';

    std::cout << "OK\n";
}

}  // namespace test

void SetUp() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

int main(int argc, char *argv[]) {
    SetUp();
    if (argc > 1 && std::strcmp(argv[1], "test") == 0) {
        test::Test();
    } else {
        std::cout << Solve(InputType{std::cin});
    }
    return 0;
}
