#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

template <class Node>
struct NodeWithId {
    Node node;
    int32_t id = 0;
};

template <class Node, class NodeComparator>
class Heap {
public:
    using TemplateNodeWithId = NodeWithId<Node>;

    [[nodiscard]] size_t Size() const {
        return nodes_.size();
    }

    [[nodiscard]] TemplateNodeWithId Top() const {
        return nodes_.front();
    }

    void Insert(TemplateNodeWithId node) {
        auto node_index = static_cast<int32_t>(nodes_.size());

        if (static_cast<size_t>(node.id) >= node_id_to_node_index_map_.size()) {
            node_id_to_node_index_map_.resize(node.id + 1);
        } else if (node_id_to_node_index_map_[node.id]) {
            throw std::invalid_argument("Node with id " + std::to_string(node.id) +
                                        " was already inserted into the heap.");
        }
        node_id_to_node_index_map_[node.id] = node_index;

        nodes_.emplace_back(std::move(node));

        SiftUp(node_index);
    }

    void Remove(const TemplateNodeWithId &node) {
        RemoveById(node.id);
    }

    bool RemoveById(int32_t node_id) {
        if (static_cast<size_t>(node_id) < node_id_to_node_index_map_.size() and
            node_id_to_node_index_map_[node_id].has_value()) {
            RemoveByIndex(node_id_to_node_index_map_[node_id].value());
            return true;
        } else {
            return false;
        }
    }

    void PopTop() {
        RemoveByIndex(0);
    }

    void PopBot() {
        node_id_to_node_index_map_[nodes_.back().id].reset();
        nodes_.pop_back();
    }

private:
    std::vector<TemplateNodeWithId> nodes_;
    std::vector<std::optional<int32_t>> node_id_to_node_index_map_;
    NodeComparator comparator_;

    [[nodiscard]] bool CompareByIndex(int32_t left_node_index, int32_t right_node_index) const {
        return comparator_(nodes_[left_node_index].node, nodes_[right_node_index].node);
    }

    void SwapByNodeIndex(int32_t left_node_index, int32_t right_node_index) {
        std::swap(node_id_to_node_index_map_[nodes_[left_node_index].id],
                  node_id_to_node_index_map_[nodes_[right_node_index].id]);
        std::swap(nodes_[left_node_index], nodes_[right_node_index]);
    }

    void RemoveByIndex(int32_t node_index) {
        SwapByNodeIndex(node_index, nodes_.size() - 1);
        bool is_new_node_on_index_less_than_previous =
            CompareByIndex(node_index, nodes_.size() - 1);

        PopBot();
        if (is_new_node_on_index_less_than_previous) {
            SiftUp(node_index);
        } else {
            SiftDown(node_index);
        }
    }

    [[nodiscard]] int32_t Left(int32_t node_index) const {
        return node_index * 2 + 1;
    }

    [[nodiscard]] int32_t Right(int32_t node_index) const {
        return node_index * 2 + 2;
    }

    [[nodiscard]] int32_t Parent(int32_t node_index) const {
        return (node_index - 1) / 2;
    }

    void SiftUp(int32_t node_index) {
        while (node_index != 0 and CompareByIndex(node_index, Parent(node_index))) {
            SwapByNodeIndex(node_index, Parent(node_index));
            node_index = Parent(node_index);
        }
    }

    void SiftDown(int32_t node_index) {
        int32_t left = Left(node_index);
        int32_t right = Right(node_index);
        int32_t size = nodes_.size();

        while ((left < size and CompareByIndex(left, node_index)) or
               (right < size and CompareByIndex(right, node_index))) {

            if (right >= size or CompareByIndex(left, right)) {
                SwapByNodeIndex(node_index, left);
                node_index = left;
            } else {
                SwapByNodeIndex(node_index, right);
                node_index = right;
            }
            left = Left(node_index);
            right = Right(node_index);
        }
    }
};

namespace io {

enum class Commands { IncrementStartOfSlidingWindow, IncrementEndOfSlidingWindow };

class InputType {
public:
    int32_t n_numbers = 0;
    int32_t n_commands = 0;
    int32_t statistic = 0;
    std::vector<int32_t> numbers;
    std::vector<Commands> commands;

    InputType(int32_t statistic, std::vector<int32_t> numbers, std::vector<Commands> commands)
        : statistic{statistic}, numbers{std::move(numbers)}, commands{std::move(commands)} {
    }

    explicit InputType(std::istream &in) {
        in >> n_numbers >> n_commands >> statistic;

        numbers.reserve(n_numbers);
        for (int32_t i = 0; i < n_numbers; ++i) {
            int32_t number = 0;
            in >> number;
            numbers.push_back(number);
        }

        commands.reserve(n_commands);
        for (int32_t i = 0; i < n_commands; ++i) {
            char command = '\0';
            in >> command;
            commands.push_back(command == 'L' ? Commands::IncrementStartOfSlidingWindow
                                              : Commands::IncrementEndOfSlidingWindow);
        }
    }
};

namespace answers {

struct SizeOfWindowIsLessThanStatisticValue {
    bool operator==(const SizeOfWindowIsLessThanStatisticValue &other) const {
        return true;
    }
};

struct ValidWindow {
    int32_t kth_statistic_value = 0;

    bool operator==(const ValidWindow &other) const {
        return kth_statistic_value == other.kth_statistic_value;
    }
};

}  // namespace n_similarity_or_mirrored_similarity_classes

using Answer = std::variant<answers::SizeOfWindowIsLessThanStatisticValue, answers::ValidWindow>;

class OutputType {
public:
    OutputType() = default;

    explicit OutputType(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            if (item == -1) {
                n_similarity_or_mirrored_similarity_classes.emplace_back(answers::SizeOfWindowIsLessThanStatisticValue{});
            } else {
                n_similarity_or_mirrored_similarity_classes.emplace_back(answers::ValidWindow{item});
            }
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto item : n_similarity_or_mirrored_similarity_classes) {
            if (auto answer = std::get_if<answers::ValidWindow>(&item)) {
                out << answer->kth_statistic_value;
            } else {
                out << "-1";
            }
            out << '\n';
        }
        return out;
    }

    bool operator==(const OutputType &other) const {
        return n_similarity_or_mirrored_similarity_classes == other.n_similarity_or_mirrored_similarity_classes;
    }

    std::vector<Answer> n_similarity_or_mirrored_similarity_classes;
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

}  // namespace io

using io::InputType, io::OutputType, io::Commands, io::Answer;

class SlidingWindowSupportingKthStatistic {
public:
    explicit SlidingWindowSupportingKthStatistic(size_t statistic) : statistic_{statistic} {
    }

    void IncrementStartOfSlidingWindow() {
        max_heap_.RemoveById(window_start_);
        min_heap_.RemoveById(window_start_);
        ++window_start_;

        OnIncrementEnd();
    }

    void IncrementEndOfSlidingWindow(int32_t number) {
        max_heap_.Insert({number, window_end_});
        ++window_end_;

        OnIncrementEnd();
    }

    Answer GetKthStatisticInSlidingWindow() {
        if (max_heap_.Size() < statistic_) {
            return io::answers::SizeOfWindowIsLessThanStatisticValue{};
        } else {
            return io::answers::ValidWindow{max_heap_.Top().node};
        }
    }

    [[nodiscard]] int32_t GetWindowEnd() const {
        return window_end_;
    }

private:
    size_t statistic_ = 0;
    Heap<int32_t, std::less<>> min_heap_;
    Heap<int32_t, std::greater<>> max_heap_;
    int32_t window_start_ = 0;
    int32_t window_end_ = 0;

    void OnIncrementEnd() {
        if (max_heap_.Size() > statistic_) {
            min_heap_.Insert(max_heap_.Top());
            max_heap_.PopTop();
        }

        if (max_heap_.Size() < statistic_ and min_heap_.Size() > 0) {
            max_heap_.Insert(min_heap_.Top());
            min_heap_.PopTop();
        }
    }
};

OutputType Solve(const InputType &input) {
    SlidingWindowSupportingKthStatistic window{static_cast<size_t>(input.statistic)};
    window.IncrementEndOfSlidingWindow(input.numbers.front());

    OutputType output;

    for (auto command : input.commands) {
        if (command == Commands::IncrementStartOfSlidingWindow) {
            window.IncrementStartOfSlidingWindow();
        } else if (command == Commands::IncrementEndOfSlidingWindow) {
            window.IncrementEndOfSlidingWindow(input.numbers[window.GetWindowEnd()]);
        } else {
            throw std::invalid_argument{"Unknown command."};
        }

        output.n_similarity_or_mirrored_similarity_classes.emplace_back(window.GetKthStatisticInSlidingWindow());
    }

    return output;
}

namespace rng {

uint32_t GetSeed() {
    static std::random_device random_device;
    auto seed = random_device();
    return seed;
}

std::mt19937 &GetEngine() {
    static std::mt19937 engine(GetSeed());
    return engine;
}

}  // namespace rng

namespace test {

class WrongAnswerException : public std::exception {
public:
    explicit WrongAnswerException(std::string const &message) : message{message.data()} {
    }

    [[nodiscard]] const char *what() const noexcept override {
        return message;
    }

    const char *message;
};

void Check(const std::string &test_case, const std::string &expected) {
    std::stringstream input_stream{test_case};
    auto input = InputType{input_stream};
    auto output = Solve(input);
    auto expected_output = OutputType{expected};
    if (not(output == expected_output)) {
        std::stringstream ss;
        ss << "\nExpected:\n" << expected_output << "\nReceived:\n" << output << "\n";
        throw WrongAnswerException{ss.str()};
    }
}

InputType GenerateRandomInput(int32_t max_number, int32_t n_numbers, int32_t statistic,
                              int32_t n_commands) {
    assert(n_commands < n_numbers);
    std::uniform_int_distribution<int32_t> distribution{0, max_number};
    std::vector<int32_t> numbers(n_numbers);
    std::vector<Commands> commands(n_commands);

    for (auto &number : numbers) {
        number = distribution(rng::GetEngine());
    }

    int32_t window_size = 0;
    for (auto &command : commands) {
        if (window_size == 0) {
            command = Commands::IncrementEndOfSlidingWindow;
        } else {
            command = distribution(rng::GetEngine()) % 3 == 0
                          ? Commands::IncrementStartOfSlidingWindow
                          : Commands::IncrementEndOfSlidingWindow;
        }
        window_size += command == Commands::IncrementEndOfSlidingWindow ? 1 : -1;
    }

    return {statistic, numbers, commands};
}

OutputType BruteForceSolve(InputType input) {
    OutputType output;
    output.n_similarity_or_mirrored_similarity_classes.reserve(input.commands.size());

    int32_t left = 0;
    int32_t right = 0;

    for (auto command : input.commands) {
        if (command == Commands::IncrementStartOfSlidingWindow) {
            ++left;
        } else {
            ++right;
        }
        assert(left <= right);
        if (right - left + 1 < input.statistic) {
            output.n_similarity_or_mirrored_similarity_classes.emplace_back(io::answers::SizeOfWindowIsLessThanStatisticValue{});
        } else {
            std::vector<int32_t> slice{input.numbers.begin() + left,
                                       input.numbers.begin() + right + 1};
            std::nth_element(slice.begin(), slice.begin() + input.statistic - 1, slice.end());
            output.n_similarity_or_mirrored_similarity_classes.emplace_back(io::answers::ValidWindow{slice[input.statistic - 1]});
        }
    }

    return output;
}

void Check(const InputType &input) {
    auto output = Solve(input);
    auto brute_force_output = BruteForceSolve(input);
    if (output.n_similarity_or_mirrored_similarity_classes.size() != brute_force_output.n_similarity_or_mirrored_similarity_classes.size()) {
        throw WrongAnswerException{"Differing number of n_similarity_or_mirrored_similarity_classes."};
    } else {
        for (size_t i = 0; i < output.n_similarity_or_mirrored_similarity_classes.size(); ++i) {
            if (not(output.n_similarity_or_mirrored_similarity_classes[i] == brute_force_output.n_similarity_or_mirrored_similarity_classes[i])) {
                output = Solve(input);
                throw WrongAnswerException{"Answer " + std::to_string(i) + " does not match."};
            }
        }
    }
}

void PrintSeed() {
    std::cerr << "Seed = " << rng::GetSeed() << std::endl;
}

void Test() {
    PrintSeed();

    Check(
        "7 4 2\n"
        "4 2 1 3 6 5 7\n"
        "RRLL\n",
        "4 2 2 -1");
    Check(
        "4 6 1\n"
        "1 2 3 4\n"
        "RLRLRL\n",
        "1 2 2 3 3 4");
    Check(
        "4 0 1\n"
        "1 2 3 4\n"
        "\n",
        "");

    for (int32_t test_case = 1; test_case < 1000; test_case <<= 1) {
        auto test_case_times_hundred = test_case / 10 * 1000;
        auto input = GenerateRandomInput(10 + test_case_times_hundred, 20 + test_case_times_hundred,
                                         3 + test_case, 19 + test_case_times_hundred);
        Check(input);
    }

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
