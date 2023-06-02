#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace io {

struct InsertRequest {
    int32_t index = 0;
    int32_t value = 0;

    explicit InsertRequest(std::istream &in) {
        in >> index >> value;
    }

    InsertRequest(int32_t index, int32_t value) : index{index}, value{value} {
    }
};

struct RemoveRequest {
    int32_t index = 0;

    explicit RemoveRequest(std::istream &in) {
        in >> index;
    }
};

struct NNumbersInIntervalNotGreaterThanValueRequest {
    int32_t from = 0;
    int32_t to = 0;
    int32_t value = 0;

    explicit NNumbersInIntervalNotGreaterThanValueRequest(std::istream &in) {
        in >> from >> to >> value;
    }
};

struct NNumbersInIntervalNotGreaterThanValueResponse {
    int32_t n_numbers = 0;
};

using Request =
    std::variant<InsertRequest, RemoveRequest, NNumbersInIntervalNotGreaterThanValueRequest>;

using Response = std::variant<NNumbersInIntervalNotGreaterThanValueResponse, std::nullopt_t>;

class Input {
public:
    std::vector<int32_t> numbers;
    std::vector<Request> requests;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_numbers = 0;
        in >> n_numbers;
        numbers.resize(n_numbers);
        for (auto &number : numbers) {
            in >> number;
        }

        requests.reserve(n_numbers);
        char c = '\0';
        while (in >> std::ws >> c) {
            if (c == '+') {
                requests.emplace_back(InsertRequest{in});
            } else if (c == '-') {
                requests.emplace_back(RemoveRequest{in});
            } else if (c == '?') {
                requests.emplace_back(NNumbersInIntervalNotGreaterThanValueRequest{in});
            } else {
                throw std::invalid_argument{"Unknown request"};
            }
        }
    }
};

class Output {
public:
    std::vector<int32_t> response_values;

    Output() = default;

    explicit Output(const std::vector<Response> &responses) {
        response_values.reserve(responses.size());
        for (auto response : responses) {
            if (auto stat_response =
                    std::get_if<NNumbersInIntervalNotGreaterThanValueResponse>(&response)) {
                response_values.emplace_back(stat_response->n_numbers);
            }
        }
    }

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            response_values.emplace_back(item);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : response_values) {
            out << item << '\n';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return response_values != other.response_values;
    }
};

std::ostream &operator<<(std::ostream &os, Output const &output) {
    return output.Write(os);
}

void SetUpFastIo() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

}  // namespace io

class LinkedListArrayBlock {
public:
    template <class Iterator>
    LinkedListArrayBlock(Iterator begin, Iterator end, size_t initial_block_size,
                         std::unique_ptr<LinkedListArrayBlock> &&next_block = nullptr)
        : initial_size_{initial_block_size}, next_block_{std::move(next_block)} {

        array_slice_.reserve(initial_size_ * 2);
        sorted_slice_.reserve(initial_size_ * 2);

        auto this_end = std::min(begin + initial_size_, end);
        array_slice_ = {begin, this_end};
        sorted_slice_ = array_slice_;
        std::sort(sorted_slice_.begin(), sorted_slice_.end());

        if (this_end < end) {
            next_block_ = std::make_unique<LinkedListArrayBlock>(this_end, end, initial_size_,
                                                                 std::move(next_block_));
        }
    }

    [[nodiscard]] size_t Size() const {
        return array_slice_.size();
    }

    template <class Request, class = std::enable_if_t<std::is_same_v<Request, io::InsertRequest> or
                                                      std::is_same_v<Request, io::RemoveRequest>>>
    std::nullopt_t Response(Request request) {
        auto size = static_cast<int32_t>(Size());
        if (request.index > size or
            (request.index == size and not std::is_same_v<Request, io::InsertRequest>)) {

            request.index -= size;
            return next_block_->Response(request);
        }

        DispatchInsertOrRemoveRequest(request);

        return std::nullopt;
    }

    io::NNumbersInIntervalNotGreaterThanValueResponse Response(
        io::NNumbersInIntervalNotGreaterThanValueRequest request) const {

        auto size = static_cast<int32_t>(Size());
        int32_t n_numbers_not_greater_in_this_block = 0;

        if (IsInnerBlock(request)) {

            auto not_less_than_value_plus_one_iter =
                std::lower_bound(sorted_slice_.begin(), sorted_slice_.end(), request.value + 1);
            n_numbers_not_greater_in_this_block =
                static_cast<int32_t>(not_less_than_value_plus_one_iter - sorted_slice_.begin());
        } else if (IsEdgeBlock(request)) {

            for (auto it = std::max(request.from, 0); it < std::min(request.to + 1, size); ++it) {
                if (array_slice_[it] <= request.value) {
                    ++n_numbers_not_greater_in_this_block;
                }
            }
        }

        if (request.to >= size) {
            request.from -= size;
            request.to -= size;
            return {n_numbers_not_greater_in_this_block + next_block_->Response(request).n_numbers};
        } else {
            return {n_numbers_not_greater_in_this_block};
        }
    }

private:
    size_t initial_size_;
    std::vector<int32_t> array_slice_;
    std::vector<int32_t> sorted_slice_;
    std::unique_ptr<LinkedListArrayBlock> next_block_;

    void DispatchInsertOrRemoveRequest(io::InsertRequest request) {
        Insert(request.index, request.value);
    }

    void DispatchInsertOrRemoveRequest(io::RemoveRequest request) {
        Remove(request.index);
    }

    void Insert(int32_t index, int32_t value) {
        array_slice_.insert(array_slice_.begin() + index, value);

        auto first_not_less_than_value_iter =
            std::lower_bound(sorted_slice_.begin(), sorted_slice_.end(), value);
        sorted_slice_.insert(first_not_less_than_value_iter, value);

        if (Size() == initial_size_ * 2) {
            SplitInTwo();
        }
    }

    void Remove(int32_t index) {
        auto value = array_slice_[index];
        auto value_iter = std::lower_bound(sorted_slice_.begin(), sorted_slice_.end(), value);
        sorted_slice_.erase(value_iter);

        array_slice_.erase(array_slice_.begin() + index);
    }

    void SplitInTwo() {
        next_block_ = std::make_unique<LinkedListArrayBlock>(
            array_slice_.begin() + static_cast<int32_t>(initial_size_), array_slice_.end(),
            initial_size_, std::move(next_block_));

        array_slice_.erase(array_slice_.begin() + static_cast<int32_t>(initial_size_),
                           array_slice_.end());

        sorted_slice_ = array_slice_;
        std::sort(sorted_slice_.begin(), sorted_slice_.end());
    }

    [[nodiscard]] bool IsInnerBlock(
        io::NNumbersInIntervalNotGreaterThanValueRequest request) const {
        auto size = static_cast<int32_t>(Size());
        return request.from <= 0 and request.to >= size;
    }

    [[nodiscard]] bool IsEdgeBlock(io::NNumbersInIntervalNotGreaterThanValueRequest request) const {
        auto size = static_cast<int32_t>(Size());
        return (0 < request.from and request.from < size) or (request.to < size);
    }
};

io::Output Solve(const io::Input &input) {
    auto sqrt_decomposition_block_size =
        static_cast<size_t>(std::sqrt((input.requests.size() + input.numbers.size()) / 2) + 1);

    LinkedListArrayBlock block{input.numbers.begin(), input.numbers.end(),
                               sqrt_decomposition_block_size};

    std::vector<io::Response> responses;
    responses.reserve(input.requests.size());

    for (auto request : input.requests) {
        auto visitor = [&block](auto request) -> io::Response { return block.Response(request); };
        responses.emplace_back(std::visit(visitor, request));
    }

    return io::Output{responses};
}

namespace test {

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

namespace detail {

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

int64_t Timeit(const std::function<void()> &function) {
    detail::TimeItInMilliseconds time;
    function();
    time.End();
    return time.Duration();
}

struct Stats {
    double mean = 0;
    double std = 0;
    double max = 0;
};

std::ostream &operator<<(std::ostream &os, const Stats &stats) {
    os << "\tMean:\t" + std::to_string(stats.mean) << '\n'
       << "\tStd:\t" + std::to_string(stats.std) << '\n'
       << "\tMax:\t" + std::to_string(stats.max) << '\n';
    return os;
}

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

}  // namespace detail

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

struct TestIo {
    io::Input input;
    std::optional<io::Output> optional_expected_output;

    explicit TestIo(io::Input input) : input{std::move(input)} {
    }

    TestIo(io::Input input, io::Output output)
        : input{std::move(input)}, optional_expected_output{std::move(output)} {
    }
};

io::Output BruteForceSolve(const io::Input &input) {
    auto numbers = input.numbers;
    io::Output output;

    for (auto request : input.requests) {
        if (std::holds_alternative<io::InsertRequest>(request)) {
            auto add = std::get<io::InsertRequest>(request);
            numbers.insert(numbers.begin() + add.index, add.value);
        } else if (std::holds_alternative<io::RemoveRequest>(request)) {
            auto remove = std::get<io::RemoveRequest>(request);
            numbers.erase(numbers.begin() + remove.index);
        } else {
            auto stat = std::get<io::NNumbersInIntervalNotGreaterThanValueRequest>(request);
            int32_t count = 0;
            for (auto i = stat.from; i < stat.to + 1; ++i) {
                if (numbers[i] <= stat.value) {
                    ++count;
                }
            }
            output.response_values.push_back(count);
        }
    }

    return output;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_numbers = 1 + test_case_id;
    int32_t n_requests = 1 + test_case_id;

    auto engine = *rng::GetEngine();
    std::uniform_int_distribution number_dist{-n_numbers, n_numbers};
    std::stringstream ss;
    ss << n_numbers << '\n';
    for (int32_t i = 0; i < n_numbers; ++i) {
        ss << number_dist(engine) << ' ';
    }
    ss << '\n';

    auto size = n_requests;
    for (int32_t index = 0; index < n_requests; ++index) {
        auto random = number_dist(engine);
        if (random % 3 == 0) {
            ss << "+ " << (random % (size + 1) + size + 1) % (size + 1) << ' ' << random << '\n';
            ++size;
        } else if (random % 3 == 1) {
            ss << "- " << (random % size + size) % size << '\n';
            --size;
        } else {
            auto [from, to] = std::minmax((random % size + size) % size,
                                          (number_dist(engine) % size + size) % size);
            ss << "? " << from << ' ' << to << ' ' << random << '\n';
        }
    }
    return TestIo{io::Input{ss}};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(100'000);
}

class TimedChecker {
public:
    std::vector<int64_t> durations;

    void Check(const std::string &test_case, const std::string &expected) {
        std::stringstream input_stream{test_case};
        io::Input input{input_stream};
        io::Output expected_output{expected};
        TestIo test_io{input, expected_output};
        Check(test_io);
    }

    io::Output TimedSolve(const io::Input &input) {
        io::Output output;
        auto solve = [&output, &input]() { output = Solve(input); };

        durations.emplace_back(detail::Timeit(solve));
        return output;
    }

    void Check(TestIo test_io) {
        auto output = TimedSolve(test_io.input);

        if (not test_io.optional_expected_output) {
            try {
                test_io.optional_expected_output = BruteForceSolve(test_io.input);
            } catch (const NotImplementedError &e) {
            }
        }

        if (test_io.optional_expected_output) {
            auto &expected_output = test_io.optional_expected_output.value();

            if (output != expected_output) {
                Solve(test_io.input);

                std::stringstream ss;
                ss << "\n================================Expected================================\n"
                   << expected_output
                   << "\n================================Received================================\n"
                   << output << "\n";

                throw WrongAnswerException{ss.str()};
            }
        }
    }
};

std::ostream &operator<<(std::ostream &os, TimedChecker &timed_checker) {
    if (not timed_checker.durations.empty()) {
        auto duration_stats =
            detail::ComputeStats(timed_checker.durations.begin(), timed_checker.durations.end());
        std::cerr << duration_stats;
        timed_checker.durations.clear();
    }
    return os;
}

void Test() {
    rng::PrintSeed();

    TimedChecker timed_checker;

    timed_checker.Check(
        "10\n"
        "455184306 359222813 948543704 914773487 861885581 253523 770029097 193773919 "
        "581789266 457415808\n"
        "- 1\n"
        "? 2 5 527021001\n"
        "? 0 5 490779085\n"
        "? 0 5 722862778\n"
        "+ 9 448694272\n"
        "- 5\n"
        "? 1 2 285404014\n"
        "- 4\n"
        "? 3 4 993634734\n"
        "+ 0 414639071",
        "1 2 2 0 2");

    std::cerr << "Basic tests OK:\n" << timed_checker;

    int32_t n_random_test_cases = 100;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
            timed_checker.Check(GenerateRandomTestIo(test_case_id));
        }

        std::cerr << "Random tests OK:\n" << timed_checker;
    } catch (const NotImplementedError &e) {
    }

    int32_t n_stress_test_cases = 1;

    try {
        for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
            timed_checker.Check(GenerateStressTestIo(test_case_id));
        }

        std::cerr << "Stress tests tests OK:\n" << timed_checker;
    } catch (const NotImplementedError &e) {
    }

    std::cerr << "OK\n";
}

}  // namespace test

int main(int argc, char *argv[]) {

    io::SetUpFastIo();

    if (argc > 1 && std::strcmp(argv[1], "test") == 0) {
        test::Test();
    } else {
        std::cout << Solve(io::Input{std::cin});
    }

    return 0;
}
