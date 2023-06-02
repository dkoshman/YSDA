#include <algorithm>
#include <array>
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
#include <utility>
#include <vector>

namespace io {

struct Request {
    enum Type { Search, Extend };

    Type type;
    std::string string;
};

class OnlineAgent {
public:
    explicit OnlineAgent(std::istream &in = std::cin, std::ostream &out = std::cout)
        : in_{in}, out_{out} {
    }

    std::optional<Request> NextRequest() {
        char type = '\0';
        std::string string;

        if (in_ >> type >> string) {
            for (auto &c : string) {
                c = static_cast<char>(std::tolower(c));
            }
            switch (type) {
                case '?':
                    return Request{Request::Search, string};
                case 'A':
                    return Request{Request::Extend, string};
                default:
                    throw std::invalid_argument{"Unknown request type."};
            }
        }
        return std::nullopt;
    }

    void Respond(bool has_found_word) {
        out_ << (has_found_word ? "YES" : "NO") << '\n';
    }

private:
    std::istream &in_;
    std::ostream &out_;
};

class Input {
public:
    std::string string;

private:
    std::shared_ptr<std::istream> in_;

public:
    std::istream *in = nullptr;

    Input() = default;

    explicit Input(std::istream &in) : in{&in} {
    }

    explicit Input(std::string string)
        : string{std::move(string)},
          in_{std::make_shared<std::stringstream>(this->string)},
          in{in_.get()} {
    }
};

class Output {
public:
    std::string n_different_substrings;

    Output() = default;

    explicit Output(std::string string) : n_different_substrings{std::move(string)} {
    }

    std::ostream &Write(std::ostream &out) const {
        out << n_different_substrings;
        return out;
    }

    bool operator!=(const Output &other) const {
        return n_different_substrings != other.n_different_substrings;
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

class Edge {
public:
    int32_t text_interval_begin = 0;
    int32_t text_interval_end = INT32_MAX;
    std::optional<int32_t> child_node;

    Edge() = default;

    explicit Edge(int32_t begin) : text_interval_begin{begin} {
    }

    Edge(int32_t begin, int32_t end, int32_t target_node)
        : text_interval_begin{begin}, text_interval_end{end}, child_node{target_node} {
    }

    [[nodiscard]] int32_t Length() const {
        return text_interval_end - text_interval_begin;
    }

    [[nodiscard]] bool IsLeaf() const {
        return not child_node;
    }

    Edge SplitChildEdge(int32_t child_edge_length, int32_t new_child_node) {
        auto child_edge = *this;
        child_edge.text_interval_begin += child_edge_length;

        text_interval_end = child_edge.text_interval_begin;
        child_node = new_child_node;

        return child_edge;
    }
};

struct Node {
    std::map<char, Edge> edges;
    std::optional<int32_t> link;

    Node() : link{0} {
    }

    Node(char letter, int32_t letter_position) : edges{{letter, Edge{letter_position}}} {
    }
};

struct Location {
    int32_t node;
    int32_t delta = 0;
    std::optional<char> first_edge_letter;

    explicit Location(int32_t node) : node{node} {
    }

    [[nodiscard]] bool IsExplicit() const {
        return delta == 0;
    }

    [[nodiscard]] bool IsRoot() const {
        return node == 0;
    }
};

class SuffixTree {
public:
    explicit SuffixTree(size_t capacity) : location_{0} {
        text_.reserve(capacity);
        nodes_.reserve(capacity);
        nodes_.emplace_back();
    }

    [[nodiscard]] bool Search(std::string word) const {
        std::optional<int32_t> node = 0;

        for (auto word_iter = word.begin(); word_iter != word.end();) {

            auto search = nodes_[node.value()].edges.find(*word_iter);
            if (search == nodes_[node.value()].edges.end()) {
                return false;
            }

            Edge edge = search->second;
            for (auto slice = edge.text_interval_begin;
                 slice < edge.text_interval_end and word_iter < word.end(); ++slice, ++word_iter) {
                if (text_[slice] != *word_iter) {
                    return false;
                }
            }
            node = edge.child_node;
        }
        return true;
    }

    void AppendLetter(char letter) {
        text_ += letter;
        std::optional<int32_t> suffix_link_from;

        while (not LocationHasLastLetter()) {
            Location previous_location = location_;

            AddLastLetterAtLocation();

            if (suffix_link_from) {
                nodes_[suffix_link_from.value()].link = location_.node;
            }
            suffix_link_from = location_.node;

            TraverseSuffixLink(previous_location);
        }

        if (suffix_link_from) {
            nodes_[suffix_link_from.value()].link = location_.node;
        }

        DescendByLastLetter();
    }

private:
    std::string text_;
    std::vector<Node> nodes_;
    Location location_;

    Edge GetImplicitEdge() {
        return nodes_[location_.node].edges[location_.first_edge_letter.value()];
    }

    char GetNextImplicitLetter() {
        return text_[GetImplicitEdge().text_interval_begin + location_.delta];
    }

    bool IsFirstImplicitNodeOnTheEdgeALeaf(Edge edge) {
        return edge.text_interval_begin + 1 == static_cast<int32_t>(text_.size());
    }

    bool LocationHasLastLetter() {
        char last_letter = text_.back();
        if (location_.IsExplicit()) {
            return nodes_[location_.node].edges.count(last_letter);
        } else {
            return GetNextImplicitLetter() == last_letter;
        }
    }

    void DescendByLastLetter() {
        if (location_.IsExplicit()) {

            location_.first_edge_letter = text_.back();
            Edge edge_to_descend = GetImplicitEdge();

            if (IsFirstImplicitNodeOnTheEdgeALeaf(edge_to_descend)) {
                return;
            } else if (not edge_to_descend.IsLeaf() and edge_to_descend.Length() == 1) {
                location_.node = edge_to_descend.child_node.value();
            } else {
                location_.delta = 1;
            }
        } else {

            Edge edge_to_descend = GetImplicitEdge();

            if (not edge_to_descend.IsLeaf() and location_.delta + 1 == edge_to_descend.Length()) {
                location_ = Location{edge_to_descend.child_node.value()};
            } else {
                ++location_.delta;
            }
        }
    }

    void AddLastLetterAtLocation() {
        if (location_.IsExplicit()) {
            nodes_[location_.node].edges.emplace(text_.back(), text_.size() - 1);
            return;
        }

        auto new_node = static_cast<int32_t>(nodes_.size());
        auto implicit_edge = GetImplicitEdge();

        nodes_.emplace_back(text_.back(), text_.size() - 1);
        Edge edge_lower_half = implicit_edge.SplitChildEdge(location_.delta, new_node);
        nodes_[location_.node].edges[location_.first_edge_letter.value()] = implicit_edge;
        nodes_[new_node].edges[GetNextImplicitLetter()] = edge_lower_half;

        location_ = Location{new_node};
    }

    void TraverseSuffixLink(Location previous_location) {
        if (location_.node == previous_location.node) {
            location_.node = nodes_[location_.node].link.value();
            return;
        }

        Edge previous_edge =
            nodes_[previous_location.node].edges[previous_location.first_edge_letter.value()];
        location_ = previous_location;

        if (location_.IsRoot()) {
            ++previous_edge.text_interval_begin;
            location_.first_edge_letter = text_[previous_edge.text_interval_begin];
            --location_.delta;
        } else {
            location_.node = nodes_[location_.node].link.value();
        }

        Edge implicit_edge = GetImplicitEdge();

        while (not implicit_edge.IsLeaf() and implicit_edge.Length() <= location_.delta) {

            previous_edge.text_interval_begin += implicit_edge.Length();
            location_.delta -= implicit_edge.Length();
            location_.node = implicit_edge.child_node.value();
            location_.first_edge_letter = text_[previous_edge.text_interval_begin];
            implicit_edge = GetImplicitEdge();
        }
    }
};

void OnlineSolve(std::istream &in, std::ostream &out) {
    io::OnlineAgent agent{in, out};
    SuffixTree tree(20'000);

    while (auto request = agent.NextRequest()) {
        switch (request->type) {

            case io::Request::Search:
                agent.Respond(tree.Search(request->string));
                break;

            case io::Request::Extend:
                for (auto letter : request->string) {
                    tree.AppendLetter(letter);
                }
                break;
        }
    }
}

io::Output Solve(const io::Input &input) {
    std::stringstream out;
    OnlineSolve(*input.in, out);
    return io::Output{out.str()};
}

namespace test {

namespace rng {

uint32_t GetSeed() {
    auto random_device = std::random_device{};
    static auto seed = random_device();
    return 3479230525;
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
    std::stringstream in{input.string};
    std::stringstream out;
    io::OnlineAgent agent{in, out};
    std::string text;

    while (auto request = agent.NextRequest()) {
        if (request->type == io::Request::Search) {
            agent.Respond(text.find(request->string) != std::string::npos);
        } else {
            text += request->string;
        }
    }

    return io::Output{out.str()};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    std::uniform_int_distribution<int32_t> coin{0, 1};
    std::uniform_int_distribution<int32_t> request_count_dist{0, 100};
    std::uniform_int_distribution<int32_t> request_type_dist{1, 100};
    std::uniform_int_distribution<char> upper_letter_dist{'A', 'B'};
    std::uniform_int_distribution<int32_t> word_length_dist{1, 10};
    std::uniform_int_distribution<int32_t> text_length_dist{1, 100};

    auto &engine = *rng::GetEngine();
    auto request_count = request_count_dist(engine);
    std::stringstream ss;
    for (int32_t request_id = 0; request_id < request_count; ++request_id) {
        if (request_id == 1 or request_type_dist(engine) < 15) {
            ss << "A ";
            auto text_length = text_length_dist(engine);
            for (int32_t i = 0; i < text_length; ++i) {
                auto letter = upper_letter_dist(engine);
                if (coin(engine)) {
                    letter = static_cast<char>(std::tolower(letter));
                }
                ss << letter;
            }
            ss << '\n';
        } else {
            ss << "? ";
            auto word_length = word_length_dist(engine);
            for (int32_t i = 0; i < word_length; ++i) {
                auto letter = upper_letter_dist(engine);
                if (coin(engine)) {
                    letter = static_cast<char>(std::tolower(letter));
                }
                ss << letter;
            }
            ss << '\n';
        }
    }

    return TestIo{io::Input{ss.str()}};
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

    TimedChecker timed_check;

    timed_check.Check(
        "? love\n"
        "? is\n"
        "A Loveis\n"
        "? love\n"
        "? WHO\n"
        "A Whoareyou\n"
        "? is\n",
        "NO\n"
        "NO\n"
        "YES\n"
        "NO\n"
        "YES\n");

    std::cerr << "Basic tests OK:\n" << timed_check;

    int32_t n_random_test_cases = 100;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
            timed_check.Check(GenerateRandomTestIo(test_case_id));
        }

        std::cerr << "Random tests OK:\n" << timed_check;
    } catch (const NotImplementedError &e) {
    }

    int32_t n_stress_test_cases = 1;

    try {
        for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
            timed_check.Check(GenerateStressTestIo(test_case_id));
        }

        std::cerr << "Stress tests tests OK:\n" << timed_check;
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
        OnlineSolve(std::cin, std::cout);
    }

    return 0;
}
