#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

struct Player {
    explicit Player(int32_t id = 0, int32_t efficiency = 0) : id{id}, efficiency{efficiency} {
    }
    int32_t id;
    int32_t efficiency;
};

using Team = std::vector<Player>;

int64_t SumUpTeamEfficiency(const Team &team) {
    int64_t sum = 0;
    for (const auto &player : team) {
        sum += player.efficiency;
    }
    return sum;
}

struct TeamRange {
    Team::const_iterator begin;
    Team::const_iterator end;
    int64_t total_efficiency = 0;
};

class InputType {
public:
    explicit InputType(std::istream &in) {
        in >> n_players;
        team.reserve(n_players);
        for (int32_t i = 0; i < n_players; ++i) {
            int32_t player_efficiency;
            in >> player_efficiency;
            team.emplace_back(i, player_efficiency);
        }
    }

    int32_t n_players = 0;
    Team team;
};

class OutputType {
public:
    OutputType() = default;

    explicit OutputType(const std::string &string) {
        std::stringstream ss{string};
        ss >> total_efficiency;
        int32_t item = 0;
        while (ss >> item) {
            player_ids.push_back(item);
        }
    }

    explicit OutputType(const Team &most_effective_cohesive_team) {
        total_efficiency = SumUpTeamEfficiency(most_effective_cohesive_team);
        for (auto &player : most_effective_cohesive_team) {
            player_ids.push_back(player.id);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        out << total_efficiency << '\n';
        for (auto item : player_ids) {
            out << item + 1 << ' ';
        }
        return out;
    }

    bool operator==(const OutputType &other) const {
        return total_efficiency == other.total_efficiency;
    }

    int64_t total_efficiency = 0;
    std::vector<int32_t> player_ids;
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

class ComparatorPlayerId {
public:
    bool operator()(const Player &lhv, const Player &rhv) const {
        return lhv.id < rhv.id;
    }
};

class ComparatorPlayerEfficiency {
public:
    bool operator()(const Player &lhv, const Player &rhv) const {
        return lhv.efficiency < rhv.efficiency;
    }
};

template <class Iterator, class Comparator>
Iterator PickPivot(Iterator begin, Iterator end, Comparator comparator) {
    Iterator mid = begin + (end - begin) / 2;
    --end;
    if (comparator(*mid, *begin)) {
        std::swap(mid, begin);
    }
    if (comparator(*mid, *end)) {
        return mid;
    }
    if (comparator(*begin, *end)) {
        return end;
    }
    return begin;
}

template <class Iterator, class Comparator>
Iterator Partition(Iterator begin, Iterator end, Iterator pivot, Comparator comparator) {
    auto pivot_value = *pivot;
    Iterator former_begin = begin;
    --end;

    while (begin <= end) {
        if (comparator(*begin, pivot_value)) {
            ++begin;
        } else if (comparator(pivot_value, *end)) {
            --end;
        } else {
            std::iter_swap(begin, end);
            ++begin;
            --end;
        }
    }

    if (begin == end + 2 and begin - 1 != former_begin) {
        return begin - 1;
    }
    return begin;
}

template <class Iterator, class Comparator>
void Sort(Iterator begin, Iterator end, Comparator comparator = ComparatorPlayerId()) {
    while (end - begin > 1) {

        Iterator pivot = PickPivot(begin, end, comparator);
        Iterator partition = Partition(begin, end, pivot, comparator);

        if (partition - begin < end - partition) {
            Sort(begin, partition, comparator);
            begin = partition;
        } else {
            Sort(partition, end, comparator);
            end = partition;
        }
    }
}

OutputType Solve(InputType input) {
    auto &whole_team = input.team;
    if (whole_team.size() <= 2) {
        return OutputType{whole_team};
    }

    Sort(whole_team.begin(), whole_team.end(), ComparatorPlayerEfficiency());

    TeamRange sliding_range{
        whole_team.begin(), whole_team.begin() + 2,
        static_cast<int64_t>(whole_team[0].efficiency) + whole_team[1].efficiency};
    TeamRange max_range = sliding_range;

    while (sliding_range.end != whole_team.end()) {
        int64_t max_possible_efficiency_for_cohesive_team =
            static_cast<int64_t>(sliding_range.begin->efficiency) +
            (sliding_range.begin + 1)->efficiency;

        while (sliding_range.end != whole_team.end() and
               sliding_range.end->efficiency <= max_possible_efficiency_for_cohesive_team) {
            sliding_range.total_efficiency += sliding_range.end->efficiency;
            ++sliding_range.end;
        }

        if (sliding_range.total_efficiency > max_range.total_efficiency) {
            max_range = sliding_range;
        }

        sliding_range.total_efficiency -= sliding_range.begin->efficiency;
        ++sliding_range.begin;
    }
    Team most_effective_cohesive_team{max_range.begin, max_range.end};
    Sort(most_effective_cohesive_team.begin(), most_effective_cohesive_team.end(),
         ComparatorPlayerId());
    return OutputType{most_effective_cohesive_team};
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
        ss << "\nExpected:\n" << expected << "\nReceived:" << output << "\n";
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
    auto begin = std::chrono::steady_clock::now();

    Check("1\n7", "7\n1");
    Check("2\n2147483647 2147483647", "4294967294\n1 2");
    Check("2\n12 21", "33\n1 2");
    Check("5\n3 2 5 4 1", "14\n1 2 3 4");
    Check("5\n1 2 4 8 16", "24\n4 5");
    Check("5\n1 1 1 1 3", "4\n1 2 3 4");
    Check("5\n1 1 1 2 3", "6\n 3 4 5");
    Check("5\n1 1 1 2 10", "12\n4 5");
    Check("5\n5 6 1 10 12", "28\n2 4 5");
    Check("5\n7 8 6 9 18", "30\n1 2 3 4");
    Check("5\n7 8 6 9 22", "31\n4 5");
    Check("5\n9 1 1 1 1", "10\n1 3");

    auto rng = std::default_random_engine{};

    for (int case_n = 0; case_n < 100; ++case_n) {
        std::vector<int32_t> test_case;
        for (unsigned i = 1; i < 1 << 21; i <<= 1) {
            for (auto j = 0; j < 1000; ++j) {
                test_case.push_back(i);
            }
        }

        std::shuffle(std::begin(test_case), std::end(test_case), rng);
        std::stringstream ss;
        ss << test_case.size() << '\n';
        for (auto i : test_case) {
            ss << ' ' << i;
        }

        Check(ss.str(), std::to_string(1000 * (1 << 20) + 1000 * (1 << 19)));
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms, ";
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
