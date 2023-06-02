#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

struct Coin {
    int32_t position;
    int32_t expire_time;
};

class InputType {
public:
    explicit InputType(std::istream &in) {
        in >> n_coins;
        coins.resize(n_coins);
        for (auto &coin : coins) {
            in >> coin.position >> coin.expire_time;
        }
    }

    int32_t n_coins = 0;
    std::vector<Coin> coins;
};

struct TimeToReachIfPossible {
    bool is_possible = true;
    int32_t time = 0;

    bool operator==(const TimeToReachIfPossible &other) const {
        if (not is_possible and not other.is_possible) {
            return true;
        }
        return is_possible == other.is_possible and time == other.time;
    }
};

TimeToReachIfPossible MinTime(const TimeToReachIfPossible &lhv, const TimeToReachIfPossible &rhv) {
    if (not lhv.is_possible) {
        return rhv;
    }
    if (not rhv.is_possible) {
        return lhv;
    }
    return lhv.time < rhv.time ? lhv : rhv;
}

class OutputType {
public:
    OutputType() = default;

    explicit OutputType(TimeToReachIfPossible min_time_to_collect_all_coins)
        : min_time_to_collect_all_coins{min_time_to_collect_all_coins} {};

    explicit OutputType(const std::string &string) {
        if (string != "No solution") {
            std::stringstream ss{string};
            ss >> min_time_to_collect_all_coins.time;
        } else {
            min_time_to_collect_all_coins.is_possible = false;
        }
    }

    std::ostream &Write(std::ostream &out) const {
        out << (min_time_to_collect_all_coins.is_possible
                    ? std::to_string(min_time_to_collect_all_coins.time)
                    : "No solution")
            << std::endl;

        return out;
    }

    bool operator==(const OutputType &other) const {
        return min_time_to_collect_all_coins == other.min_time_to_collect_all_coins;
    }

    TimeToReachIfPossible min_time_to_collect_all_coins;
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

class ComparatorCoinPositionLess {
public:
    bool operator()(const Coin &lhv, const Coin &rhv) const {
        return lhv.position < rhv.position;
    }
};

struct TimeInterval {
    TimeToReachIfPossible time_if_ali_ends_on_left;
    TimeToReachIfPossible time_if_ali_ends_on_right;
};

TimeToReachIfPossible TryToGetFromFirstToSecondInTime(const Coin &first, const Coin &second,
                                                      TimeToReachIfPossible time_passed) {
    if (not time_passed.is_possible) {
        return {false};
    }
    auto time_between = abs(second.position - first.position);
    if (second.expire_time - time_passed.time < time_between) {
        return {false};
    }
    return {true, time_passed.time + time_between};
}

OutputType Solve(InputType input) {
    auto &coins = input.coins;
    std::sort(coins.begin(), coins.end(), ComparatorCoinPositionLess());
    std::vector<TimeInterval> intervals(input.n_coins), intervals_one_longer(input.n_coins);

    for (int intervals_size = 0; intervals_size < input.n_coins; ++intervals_size) {
        for (int left_border = 0; left_border < input.n_coins - intervals_size - 1; ++left_border) {
            int right_border = left_border + intervals_size;

            auto time_on_right = MinTime(
                TryToGetFromFirstToSecondInTime(coins[left_border], coins[right_border + 1],
                                                intervals[left_border].time_if_ali_ends_on_left),
                TryToGetFromFirstToSecondInTime(coins[right_border], coins[right_border + 1],
                                                intervals[left_border].time_if_ali_ends_on_right));
            intervals_one_longer[left_border].time_if_ali_ends_on_right = time_on_right;

            auto time_on_left = MinTime(TryToGetFromFirstToSecondInTime(
                                            coins[left_border + 1], coins[left_border],
                                            intervals[left_border + 1].time_if_ali_ends_on_left),
                                        TryToGetFromFirstToSecondInTime(
                                            coins[right_border + 1], coins[left_border],
                                            intervals[left_border + 1].time_if_ali_ends_on_right));
            intervals_one_longer[left_border].time_if_ali_ends_on_left = time_on_left;
        }

        intervals.swap(intervals_one_longer);
    }

    auto min_time = MinTime(intervals_one_longer.front().time_if_ali_ends_on_left,
                            intervals_one_longer.front().time_if_ali_ends_on_right);

    return OutputType{min_time};
}

class WrongAnswerException : public std::exception {
public:
    explicit WrongAnswerException(std::string const &message) : message{message.data()} {
    }

    [[nodiscard]] const char *what() const noexcept override {
        return message;
    }

    const char *message;
};

void ThrowIfAnswerIsIncorrect(const InputType &input, const OutputType &output,
                              const OutputType &expected) {
    if (not(output == expected)) {
        std::stringstream ss;
        ss << "\nExpected:\n" << expected << "\nReceived:\n" << output << "\n";
        throw WrongAnswerException{ss.str()};
    }
}

void Check(const std::string &test_case, const std::string &expected) {
    std::stringstream input_stream{test_case};
    auto input = InputType{input_stream};
    auto output = Solve(input);
    ThrowIfAnswerIsIncorrect(input, output, OutputType{expected});
}

void Test() {
    Check(
        "5\n"
        "1 3\n"
        "3 1\n"
        "5 8\n"
        "8 19\n"
        "10 15\n",
        "11");
    Check(
        "5\n"
        "1 5\n"
        "2 1\n"
        "3 4\n"
        "4 2\n"
        "5 3\n",
        "No solution");
    Check(
        "3\n"
        "3 3\n"
        "1 1\n"
        "2 0\n",
        "3");
    Check(
        "3\n"
        "3 100000\n"
        "1 1\n"
        "2 0\n",
        "3");
    Check(
        "4\n"
        "8 0\n"
        "1 11\n"
        "9 3\n"
        "7 1\n",
        "11");
    Check(
        "5\n"
        "8 0\n"
        "1 11\n"
        "4 1000\n"
        "9 3\n"
        "7 1\n",
        "11");

    std::cout << "OK\n";
}

void SetUp() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
}

int main(int argc, char *argv[]) {
    SetUp();
    if (argc > 1 && std::strcmp(argv[1], "test") == 0) {
        Test();
    } else {
        std::cout << Solve(InputType{std::cin});
    }
    return 0;
}
