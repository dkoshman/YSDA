#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace utils {

template <class T>
T NonNegativeMod(T value, T divisor) {
    if (divisor == 0) {
        throw std::invalid_argument("Zero divisor.");
    }
    if (divisor < 0) {
        throw std::invalid_argument("Negative divisor.");
    }
    auto mod = value % divisor;
    if (mod < 0) {
        mod += divisor;
    }
    return mod;
}

template <typename Integral>
static Integral LargestPowerOfTwoNotGreaterThan(Integral value) {
    if (value <= 0) {
        throw std::invalid_argument{"Non positive logarithm argument."};
    }
    Integral log_two = 0;
    while (value >>= 1) {
        ++log_two;
    }
    return log_two;
}

template <typename Number>
Number Squared(Number value) {
    return value * value;
}

template <typename Container, typename Comparator>
class RangeMinQueryResponder {
public:
    using Value = typename Container::value_type;

    explicit RangeMinQueryResponder(const Container &container, const Comparator &comparator)
        : container_{container}, comparator_{comparator} {
    }

    void Preprocess() {
        if (not rmq_preprocessed_.empty()) {
            return;
        }

        auto log_two = LargestPowerOfTwoNotGreaterThan(container_.size());

        rmq_preprocessed_.resize(log_two + 1);
        rmq_preprocessed_.front() = container_;

        for (size_t interval_size_log_two = 1; interval_size_log_two <= log_two;
             ++interval_size_log_two) {

            auto &rmq_preprocessed_by_interval_size = rmq_preprocessed_[interval_size_log_two];
            auto interval_size = 1 << interval_size_log_two;
            auto n_valid_begin_positions = container_.size() - interval_size + 1;
            rmq_preprocessed_by_interval_size.resize(n_valid_begin_positions);

            for (size_t begin = 0; begin < n_valid_begin_positions; ++begin) {

                auto &left_half_min = MinOnPreprocessedIntervalByEnd(begin + interval_size / 2,
                                                                     interval_size_log_two - 1);
                auto &right_half_min = MinOnPreprocessedIntervalByBegin(begin + interval_size / 2,
                                                                        interval_size_log_two - 1);

                MinOnPreprocessedIntervalByBegin(begin, interval_size_log_two) =
                    std::min(left_half_min, right_half_min, comparator_);
            }
        }
    }

    Value GetRangeMin(size_t begin, size_t end) {
        Preprocess();

        auto log_two = LargestPowerOfTwoNotGreaterThan(end - begin);

        auto min_on_left_overlapping_half = MinOnPreprocessedIntervalByBegin(begin, log_two);
        auto min_on_right_overlapping_half = MinOnPreprocessedIntervalByEnd(end, log_two);

        return std::min(min_on_left_overlapping_half, min_on_right_overlapping_half, comparator_);
    }

private:
    const Container &container_;
    const Comparator &comparator_;
    std::vector<std::vector<Value>> rmq_preprocessed_;

    template <typename Integral>
    static Integral LargestPowerOfTwoNotGreaterThan(Integral value) {
        Integral log_two = 0;
        while (value >>= 1) {
            ++log_two;
        }
        return log_two;
    }

    Value &MinOnPreprocessedIntervalByBegin(size_t begin, size_t interval_size_log_two) {
        return rmq_preprocessed_[interval_size_log_two][begin];
    }

    Value &MinOnPreprocessedIntervalByEnd(size_t end, size_t interval_size_log_two) {
        return rmq_preprocessed_[interval_size_log_two][end - (1 << interval_size_log_two)];
    }
};

template <class Key, class Value>
std::vector<Value> GetMapValues(const std::map<Key, Value> &map) {
    std::vector<Value> values;
    values.reserve(map.size());
    for (auto &pair : map) {
        values.emplace_back(pair.second);
    }
    return values;
}

template <typename Iterator, typename Comparator>
std::vector<size_t> ArgSort(Iterator begin, Iterator end) {
    Comparator comparator;
    std::vector<size_t> indices(end - begin);
    std::iota(indices.begin(), indices.end(), 0);
    auto arg_comparator = [&begin, &comparator](size_t left, size_t right) -> bool {
        return comparator(*(begin + left), *(begin + right));
    };
    std::sort(indices.begin(), indices.end(), arg_comparator);
    return indices;
}

struct ComparatorLess {
    template <typename T>
    bool operator()(const T &left, const T &right) const {
        return left < right;
    }
};

template <typename Iterator>
std::vector<size_t> ArgSort(Iterator begin, Iterator end) {
    return ArgSort<Iterator, ComparatorLess>(begin, end);
}

}  // namespace utils

namespace io {

std::string BuildValidStringCharacters() {
    std::string string{'_'};
    for (auto c = '0'; c <= '9'; ++c) {
        string += c;
    }
    for (auto c = 'a'; c <= 'z'; ++c) {
        string += c;
    }
    for (auto c = 'A'; c <= 'Z'; ++c) {
        string += c;
    }
    std::sort(string.begin(), string.end());
    return string;
}

const std::string &GetValidStringCharacters() {
    auto static string = BuildValidStringCharacters();
    return string;
}

class Input {
public:
    static constexpr int32_t kMaxStringSize = 14;
    std::string string_to_find_collision_for;
    int64_t prime_mod = 0;
    int64_t prime_power = 0;

    Input() = default;

    explicit Input(std::istream &in) {
        in >> string_to_find_collision_for >> prime_mod >> prime_power;
    }

    [[nodiscard]] int64_t StringHash(const std::string &string) const {
        int64_t hash = 0;
        for (auto c : string) {
            hash = ModPrime(ProductModPrime(hash, prime_power) + c);
        }
        return hash;
    }

    [[nodiscard]] int64_t ModPrime(int64_t value) const {
        return utils::NonNegativeMod(value, prime_mod);
    }

    [[nodiscard]] int64_t ProductModPrime(int64_t first, int64_t second) const {
        if (first > INT32_MAX) {
            auto mod_halved = ProductModPrime(first / 2, second);
            return ModPrime(2 * mod_halved + (first % 2) * second);
        } else if (second > INT32_MAX) {
            return ProductModPrime(second, first);
        } else {
            return ModPrime(first * second);
        }
    }

    [[nodiscard]] int64_t RaiseToPowerModPrime(int64_t base, int64_t power) const {
        return power == 0 ? 1 : ProductModPrime(RaiseToPowerModPrime(base, power - 1), base);
    }
};

class Output {
public:
    std::string string_with_same_hash;
    std::optional<const io::Input *> input;

    Output() = default;

    explicit Output(std::string string) : string_with_same_hash{std::move(string)} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << string_with_same_hash;
        return out;
    }

    bool operator!=(const Output &other) const {
        auto maybe_input = input ? input : other.input;
        if (maybe_input) {
            return maybe_input.value()->StringHash(string_with_same_hash) !=
                   maybe_input.value()->StringHash(other.string_with_same_hash);
        } else {
            return string_with_same_hash != other.string_with_same_hash;
        }
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

class AllValidStringsOfGivenSizeIterable {
public:
    AllValidStringsOfGivenSizeIterable() = default;

    explicit AllValidStringsOfGivenSizeIterable(int32_t string_size) : string_size_{string_size} {
    }

    struct Iterator {

        Iterator() : valid_string_iter_{io::GetValidStringCharacters().end()} {
        }

        explicit Iterator(int32_t string_size) : string_position{string_size - 1} {
            string_.resize(string_size);
            string_ptr_ = &string_;
            Initialize();
        }

        Iterator(std::string *string_ptr, int32_t string_position)
            : string_position{string_position}, string_ptr_{string_ptr} {
            Initialize();
        }

        const std::string &operator*() const {
            return string_;
        }

        Iterator &operator++() {
            if (not one_shorter_iterator_) {
                ++valid_string_iter_;
            } else {
                one_shorter_iterator_->operator++();
                if (one_shorter_iterator_->IsExhausted()) {
                    ++valid_string_iter_;
                    one_shorter_iterator_->Reset();
                }
            }
            UpdateString();
            return *this;
        }

        friend bool operator==(const Iterator &left, const Iterator &right) {
            return left.IsExhausted() and right.IsExhausted();
        }

        friend bool operator!=(const Iterator &left, const Iterator &right) {
            return not operator==(left, right);
        }

    protected:
        int32_t string_position = 0;
        std::string::const_iterator valid_string_iter_ = io::GetValidStringCharacters().begin();
        const std::string::const_iterator valid_string_end_ = io::GetValidStringCharacters().end();
        std::string string_;
        std::string *string_ptr_ = nullptr;
        std::unique_ptr<Iterator> one_shorter_iterator_;

        void Initialize() {
            if (string_position > 0) {
                one_shorter_iterator_ =
                    std::make_unique<Iterator>(string_ptr_, string_position - 1);
            }
            UpdateString();
        }

        void UpdateString() {
            (*string_ptr_)[string_position] = *valid_string_iter_;
        }

        void Reset() {
            valid_string_iter_ = io::GetValidStringCharacters().begin();
        }

        [[nodiscard]] bool IsExhausted() const {
            return valid_string_iter_ == valid_string_end_;
        }
    };

    Iterator begin() {
        return Iterator{string_size_};
    }

    Iterator end() {
        return Iterator{};
    }

private:
    int32_t string_size_;
};

io::Output SolveEqualPrimesEdgeCase(const io::Input &input) {
    io::Output output;
    output.input = &input;

    if (input.string_to_find_collision_for.size() > 1) {
        output.string_with_same_hash = {input.string_to_find_collision_for.begin() + 1,
                                        input.string_to_find_collision_for.end()};
    } else {
        output.string_with_same_hash =
            input.string_to_find_collision_for + input.string_to_find_collision_for;
    }

    return output;
}

io::Output Solve(const io::Input &input) {

    if (input.prime_mod == input.prime_power) {
        return SolveEqualPrimesEdgeCase(input);
    }

    io::Output output;
    output.input = &input;
    auto string_half_size = io::Input::kMaxStringSize / 2;

    std::unordered_map<int64_t, std::string> map;
    map.reserve(io::GetValidStringCharacters().size() << string_half_size / 2);

    auto hash_to_match = input.StringHash(input.string_to_find_collision_for);
    auto prime_power_exp_half_size =
        input.RaiseToPowerModPrime(input.prime_power, string_half_size);

    for (const auto &first_string_half : AllValidStringsOfGivenSizeIterable{string_half_size}) {

        auto hash = input.StringHash(first_string_half);
        map.insert({hash, first_string_half});

        auto expected_hash_for_second_string_half =
            input.ModPrime(hash_to_match - input.ProductModPrime(prime_power_exp_half_size, hash));

        auto map_find = map.find(expected_hash_for_second_string_half);
        if (map_find != map.end()) {
            output.string_with_same_hash = first_string_half + map_find->second;
            if (output.string_with_same_hash != input.string_to_find_collision_for) {
                return output;
            }
        }
    }

    throw std::runtime_error{"No collision found."};
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
    if (input.prime_mod > 1'000) {
        throw NotImplementedError{};
    }

    auto hash_to_match = input.StringHash(input.string_to_find_collision_for);
    io::Output output;
    output.input = &input;

    for (const auto &string : AllValidStringsOfGivenSizeIterable(4)) {
        if (input.StringHash(string) == hash_to_match and
            string != input.string_to_find_collision_for) {
            output.string_with_same_hash = string;
            return output;
        }
    }

    throw std::runtime_error{"No collision found."};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t string_size = std::min(io::Input::kMaxStringSize, 1 + test_case_id / 50);
    std::array primes_power{2, 3, 5, 7, 11, 13, 17, 23, 53, 1999};
    auto prime_power = primes_power[std::min<int32_t>(primes_power.size() - 1, test_case_id / 10)];
    std::array primes_mod{11l, 247l, 547l, 1231l, 189871l, 10'000'000'019l, 99'999'999'977l};
    auto prime_mod = primes_mod[std::min<int32_t>(primes_mod.size() - 1, test_case_id / 25)];

    io::Input input;
    input.prime_mod = prime_mod;
    input.prime_power = prime_power;

    auto valid_chars = io::GetValidStringCharacters();
    std::uniform_int_distribution<size_t> char_distribution{0, valid_chars.size() - 1};

    while (input.string_to_find_collision_for.size() < static_cast<size_t>(string_size)) {
        input.string_to_find_collision_for += valid_chars[char_distribution(*rng::GetEngine())];
    }

    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(1'000);
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

    std::stringstream ss{
        "abacaba\n"
        "17239 257"};
    io::Input input{ss};
    assert(input.StringHash("abacaba") == input.StringHash("J2jLwXRJCM"));

    timed_checker.Check(
        "abacaba\n"
        "17239 257",
        "J2jLwXRJCM");

    std::cerr << "Basic tests OK:\n" << timed_checker;

    int32_t n_random_test_cases = 100;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
            timed_checker.Check(GenerateRandomTestIo(test_case_id));
        }

        std::cerr << "Random tests OK:\n" << timed_checker;
    } catch (const NotImplementedError &e) {
    }

    int32_t n_stress_test_cases = 10;

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
