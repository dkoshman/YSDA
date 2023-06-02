#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <deque>
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

namespace utils {
template <class Dividend, class Divisor>
Divisor NonNegativeMod(Dividend value, Divisor divisor) {
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
    return static_cast<Divisor>(mod);
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
Number PowerOfTwo(Number value) {
    return value * value;
}

}  // namespace utils

namespace io {

using Matrix = std::vector<std::vector<int32_t>>;

class Input {
public:
    Matrix matrix;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_rows = 0;
        int32_t n_columns = 0;
        in >> n_rows >> n_columns;

        matrix.resize(n_rows);
        for (auto &row : matrix) {
            row.resize(n_columns);
            for (auto &element : row) {
                char c = '\0';
                in >> c;
                element = c - 'a';
            }
        }
    }
};

struct Point {
    int32_t xx = 0;
    int32_t yy = 0;
};

bool operator==(const Point &left, const Point &right) {
    return left.xx == right.xx and left.yy == right.yy;
}

bool operator!=(const Point &left, const Point &right) {
    return not(left == right);
}

class Output {
public:
    int32_t size_of_largest_equal_subsquares = 0;
    std::optional<Point> first_subsquare_corner_closest_to_origin;
    std::optional<Point> second_subsquare_corner_closest_to_origin;

    Output() = default;

    explicit Output(int32_t size) : size_of_largest_equal_subsquares{size} {
    }

    explicit Output(int32_t size, Point upper_left, Point lower_right)
        : size_of_largest_equal_subsquares{size},
          first_subsquare_corner_closest_to_origin{upper_left},
          second_subsquare_corner_closest_to_origin{lower_right} {
    }

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        ss >> size_of_largest_equal_subsquares;
        if (size_of_largest_equal_subsquares > 0) {
            for (auto corner : {&first_subsquare_corner_closest_to_origin,
                                &second_subsquare_corner_closest_to_origin}) {
                int32_t xx = 0;
                int32_t yy = 0;
                ss >> xx >> yy;
                *corner = Point{xx, yy};
            }
        }
    }

    std::ostream &Write(std::ostream &out) const {
        out << size_of_largest_equal_subsquares << '\n';
        for (auto corner : {first_subsquare_corner_closest_to_origin,
                            second_subsquare_corner_closest_to_origin}) {
            if (corner) {
                out << corner->xx + 1 << ' ' << corner->yy + 1 << '\n';
            }
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return size_of_largest_equal_subsquares != other.size_of_largest_equal_subsquares;
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

using io::Point;

class AdditiveSquareConvolutionHasher {
public:
    const int32_t max_side = 0;
    const int32_t prime = 0;

    AdditiveSquareConvolutionHasher(int32_t max_square_side, int32_t prime)
        : max_side{max_square_side}, prime{prime}, kernel_(max_square_side + 1) {

        std::uniform_int_distribution<int64_t> distribution{2, prime - 1};
        seed_parameter_ = distribution(*rng::GetEngine());
        int32_t seed_parameter_to_power_n = 1;

        for (auto &kernel_row : kernel_) {
            kernel_row.resize(max_square_side + 1);
            for (auto &kernel_element : kernel_row) {

                kernel_element = seed_parameter_to_power_n;
                seed_parameter_to_power_n = Mod(seed_parameter_ * seed_parameter_to_power_n);
            }
            std::reverse(kernel_row.begin(), kernel_row.end());
        }
        std::reverse(kernel_.begin(), kernel_.end());
    }

    [[nodiscard]] int32_t ComputeConvolutionHash(const io::Matrix &matrix,
                                                 Point corner_closest_to_origin,
                                                 int32_t side) const {
        auto &cc = corner_closest_to_origin;
        int64_t hash = 0;
        for (int32_t xx = 0; xx < side; ++xx) {
            for (int32_t yy = 0; yy < side; ++yy) {
                hash += matrix[xx + cc.xx][yy + cc.yy] * kernel_[xx][yy];
            }
        }
        return Mod(hash);
    }

    [[nodiscard]] int32_t HashOfGreaterXNeighborSquare(const io::Matrix &matrix,
                                                       Point corner_closest_to_origin, int32_t side,
                                                       int64_t hash) const {
        auto &cc = corner_closest_to_origin;

        for (int32_t yy = 0; yy < side; ++yy) {
            hash += -kernel_[0][yy] * matrix[cc.xx][cc.yy + yy] +
                    kernel_[side][yy] * matrix[cc.xx + side][cc.yy + yy];
        }

        return Mod(hash * GetSeedParameterToThePower(1));
    }

    [[nodiscard]] int32_t HashOfGreaterYNeighborSquare(const io::Matrix &matrix,
                                                       Point corner_closest_to_origin, int32_t side,
                                                       int64_t hash) const {
        auto &cc = corner_closest_to_origin;

        for (int32_t xx = 0; xx < side; ++xx) {
            hash += -kernel_[xx][0] * matrix[cc.xx + xx][cc.yy] +
                    kernel_[xx][side] * matrix[cc.xx + xx][cc.yy + side];
        }

        return Mod(hash * GetSeedParameterToThePower(max_side));
    }

private:
    int64_t seed_parameter_ = 0;
    std::vector<std::vector<int32_t>> kernel_;

    [[nodiscard]] int32_t Mod(int64_t value) const {
        return utils::NonNegativeMod(value, prime);
    }

    [[nodiscard]] int32_t GetSeedParameterToThePower(int32_t power) const {
        auto quotient = power / max_side;
        auto remainder = power % max_side;
        return kernel_[max_side - remainder][max_side - quotient];
    }
};

class MatrixSquare {
public:
    const io::Matrix *matrix;
    Point corner_closest_to_origin;
    int32_t side = 0;
    int32_t hash = 0;

    MatrixSquare(const io::Matrix &matrix, const AdditiveSquareConvolutionHasher &hasher,
                 Point corner_closest_to_origin, int32_t side)
        : matrix{&matrix},
          corner_closest_to_origin{corner_closest_to_origin},
          side{side},
          hash{hasher.ComputeConvolutionHash(matrix, corner_closest_to_origin, side)} {
    }

    bool operator==(const MatrixSquare &other) const {
        if (side != other.side or corner_closest_to_origin == other.corner_closest_to_origin) {
            return false;
        }

        auto &cc = corner_closest_to_origin;
        auto &other_cc = other.corner_closest_to_origin;

        for (int32_t xx = 0; xx < side; ++xx) {
            for (int32_t yy = 0; yy < side; ++yy) {
                if ((*matrix)[cc.xx + xx][cc.yy + yy] !=
                    (*other.matrix)[other_cc.xx + xx][other_cc.yy + yy]) {
                    return false;
                }
            }
        }

        return true;
    }

    bool TryIncrementXCoordinate(const AdditiveSquareConvolutionHasher &hasher) {
        if (corner_closest_to_origin.xx + side >= static_cast<int32_t>(matrix->size())) {
            return false;
        } else {
            hash =
                hasher.HashOfGreaterXNeighborSquare(*matrix, corner_closest_to_origin, side, hash);
            ++corner_closest_to_origin.xx;
            return true;
        }
    }

    bool TryIncrementYCoordinate(const AdditiveSquareConvolutionHasher &hasher) {
        if (corner_closest_to_origin.yy + side >= static_cast<int32_t>(matrix->front().size())) {
            return false;
        } else {
            hash =
                hasher.HashOfGreaterYNeighborSquare(*matrix, corner_closest_to_origin, side, hash);
            ++corner_closest_to_origin.yy;
            return true;
        }
    }
};

template <class Value, class HashValue = int32_t>
class HashTable {
public:
    std::vector<std::vector<Value>> table;

    explicit HashTable(int32_t capacity) : table(capacity) {
    }

    void Clear() {
        for (auto &bucket : table) {
            bucket.clear();
        }
    }

    [[nodiscard]] bool HasValue(const Value &value, HashValue hash) const {
        auto is_equal = [&value](const Value &other) -> bool { return value == other; };
        return std::any_of(table[hash].begin(), table[hash].end(), is_equal);
    }

    [[nodiscard]] std::optional<Value> GetEqualTo(const Value &value, HashValue hash) const {
        for (auto &other : table[hash]) {
            if (value == other) {
                return other;
            }
        }
        return std::nullopt;
    }

    void Insert(const Value &value, HashValue hash) {
        if (not HasValue(value, hash)) {
            table[hash].emplace_back(value);
        }
    }
};

io::Output IsThereEqualMatrixSquaresWithGivenSideSize(const io::Matrix &matrix,
                                                      const AdditiveSquareConvolutionHasher &hasher,
                                                      HashTable<MatrixSquare> *hash_table,
                                                      int32_t side) {
    hash_table->Clear();
    MatrixSquare least_y_square{matrix, hasher, {0, 0}, side};

    do {
        auto matrix_square = least_y_square;
        do {

            auto maybe_equal_square = hash_table->GetEqualTo(matrix_square, matrix_square.hash);
            if (maybe_equal_square) {
                return io::Output{matrix_square.side, matrix_square.corner_closest_to_origin,
                                  maybe_equal_square->corner_closest_to_origin};
            }

            hash_table->Insert(matrix_square, matrix_square.hash);
        } while (matrix_square.TryIncrementYCoordinate(hasher));
    } while (least_y_square.TryIncrementXCoordinate(hasher));

    return io::Output{0};
}

io::Output Solve(const io::Input &input) {
    int32_t square_side_low = 0;
    auto square_side_high =
        static_cast<int32_t>(std::min(input.matrix.size(), input.matrix.front().size()) + 1);

    int32_t prime = 1'000'003;
    auto hasher = AdditiveSquareConvolutionHasher{square_side_high, prime};
    HashTable<MatrixSquare> hash_table{prime};

    io::Output output{0};
    while (square_side_low + 1 < square_side_high) {

        auto mid = (square_side_low + square_side_high) / 2;

        auto better_output =
            IsThereEqualMatrixSquaresWithGivenSideSize(input.matrix, hasher, &hash_table, mid);

        if (better_output.size_of_largest_equal_subsquares == 0) {
            square_side_high = mid;
        } else {
            output = better_output;
            square_side_low = mid;
        }
    }

    return output;
}

namespace test {

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

void NextPointInSquare(Point square_upper_left, int32_t square_side, Point *point) {
    if (point->xx == square_upper_left.xx + square_side - 1) {
        point->xx = square_upper_left.xx;
        ++point->yy;
    } else {
        ++point->xx;
    }
}

Point SquareEndPoint(Point square_upper_left, int32_t square_side) {
    return {square_upper_left.xx, square_upper_left.yy + square_side};
}

bool AreSubSquaresEqual(const io::Matrix &matrix, int32_t square_side, Point first, Point second) {

    for (Point first_begin{first}, second_begin{second};
         first != SquareEndPoint(first_begin, square_side);
         NextPointInSquare(first_begin, square_side, &first),
         NextPointInSquare(second_begin, square_side, &second)) {

        if (matrix[first.xx][first.yy] != matrix[second.xx][second.yy]) {
            return false;
        }
    }

    return true;
}

io::Output BruteForceSolve(const io::Input &input) {
    auto size = static_cast<int32_t>(input.matrix.size());

    assert(input.matrix.size() == input.matrix.front().size());

    io::Output answer{0};
    Point begin{0, 0};

    for (auto first = begin; first != SquareEndPoint(begin, size);
         NextPointInSquare(begin, size, &first)) {
        for (auto second = begin; second != SquareEndPoint(begin, size);
             NextPointInSquare(begin, size, &second)) {

            if (first == second) {
                continue;
            }

            for (int32_t side = size - std::max({first.xx, second.xx, first.yy, second.yy});
                 side > answer.size_of_largest_equal_subsquares; --side) {

                if (AreSubSquaresEqual(input.matrix, side, first, second)) {
                    answer = io::Output{side, first, second};
                }
            }
        }
    }

    return answer;
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t square_matrix_side = 1 + test_case_id / 10;
    std::uniform_int_distribution<char> letters_dist{'a', 'b'};

    std::stringstream ss;
    ss << square_matrix_side << ' ' << square_matrix_side << '\n';
    for (int32_t row = 0; row < square_matrix_side; ++row) {
        for (int32_t column = 0; column < square_matrix_side; ++column) {
            ss << letters_dist(*rng::GetEngine());
        }
        ss << '\n';
    }

    io::Input input{ss};
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    throw NotImplementedError{};
    io::Input input;
    return TestIo{input};
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
        "5 10\n"
        "ljkfghdfas\n"
        "isdfjksiye\n"
        "pgljkijlgp\n"
        "eyisdafdsi\n"
        "lnpglkfkjl\n",
        "3\n"
        "0 0"
        "2 2");

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
