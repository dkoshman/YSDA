// https://contest.yandex.ru/contest/29061/problems/

#include <algorithm>
#include <array>
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
#include <variant>
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

struct MakeFirstAParentOfSecond {
    struct Request {
        int32_t first = 0;
        int32_t second = 0;

        Request(int32_t first, int32_t second) : first{first}, second{second} {
        }
    };

    struct Response {
        bool second_didnt_have_a_parent_before = false;

        explicit Response(bool second_didnt_have_a_parent_before)
            : second_didnt_have_a_parent_before{second_didnt_have_a_parent_before} {
        }
    };
};

struct FindRootOfNode {
    struct Request {
        int32_t node = 0;

        explicit Request(int32_t node) : node{node} {
        }
    };

    struct Response {
        int32_t node_root = 0;

        explicit Response(int32_t node_root) : node_root{node_root} {
        }
    };
};

using Request = std::variant<MakeFirstAParentOfSecond::Request, FindRootOfNode::Request>;

using Response = std::variant<MakeFirstAParentOfSecond::Response, FindRootOfNode::Response>;

class Input {
public:
    std::vector<Request> requests;
    int32_t n_nodes = 0;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_requests = 0;
        in >> n_nodes >> n_requests;

        requests.reserve(n_requests);
        for (int32_t request_id = 0; request_id < n_requests; ++request_id) {
            int32_t first = 0;
            in >> first;

            --first;

            if (in.peek() == ' ') {
                int32_t second = 0;
                in >> second;

                --second;

                requests.emplace_back(MakeFirstAParentOfSecond::Request{first, second});
            } else {
                requests.emplace_back(FindRootOfNode::Request{first});
            }
        }
    }
};

class Output {
public:
    std::optional<std::vector<Response>> responses;
    std::vector<int32_t> response_values;

    Output() = default;

    explicit Output(std::vector<Response> responses) : responses{std::move(responses)} {

        response_values.reserve(this->responses.value().size());

        for (auto &response : this->responses.value()) {
            if (auto connect_response =
                    std::get_if<MakeFirstAParentOfSecond::Response>(&response)) {
                response_values.emplace_back(connect_response->second_didnt_have_a_parent_before);
            } else {
                auto root = std::get<FindRootOfNode::Response>(response).node_root;

                ++root;

                response_values.emplace_back(root);
            }
        }
    }

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t item = 0;
        while (ss >> item) {
            response_values.push_back(item);
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

}  // namespace io

using io::Input, io::Output;

class NodeWithRootLink {
public:
    explicit NodeWithRootLink(int32_t node) : node_{node} {
    }

    [[nodiscard]] bool HasDirectParent() const {
        return static_cast<bool>(direct_parent_);
    }

    void CompressPathToRoot() {
        std::vector<NodeWithRootLink *> chain_of_parents;

        auto root = this;
        while (root->last_known_root_) {
            chain_of_parents.emplace_back(root);
            root = root->last_known_root_.value();
        }

        for (auto node : chain_of_parents) {
            node->last_known_root_ = root;
        }
    }

    [[nodiscard]] int32_t GetRoot() {
        if (not HasDirectParent()) {
            return node_;
        } else {

            CompressPathToRoot();
            return last_known_root_.value()->node_;
        }
    }

    void AssignDirectParent(NodeWithRootLink &direct_parent) {
        if (HasDirectParent()) {
            throw std::runtime_error{"NodeT already has a direct parent."};
        }
        direct_parent_ = &direct_parent;
        last_known_root_ = direct_parent_;
    }

private:
    int32_t node_ = 0;
    std::optional<NodeWithRootLink *> direct_parent_;
    std::optional<NodeWithRootLink *> last_known_root_;
};

class Responder {
public:
    std::vector<NodeWithRootLink> nodes;

    explicit Responder(int32_t n_nodes) {
        nodes.reserve(n_nodes);

        for (int32_t node = 0; node < n_nodes; ++node) {
            nodes.emplace_back(node);
        }
    }

    io::MakeFirstAParentOfSecond::Response Respond(io::MakeFirstAParentOfSecond::Request request) {

        if (nodes[request.second].HasDirectParent() or
            nodes[request.first].GetRoot() == request.second) {
            return io::MakeFirstAParentOfSecond::Response{false};

        } else {
            nodes[request.second].AssignDirectParent(nodes[request.first]);
            return io::MakeFirstAParentOfSecond::Response{true};
        }
    }

    io::FindRootOfNode::Response Respond(io::FindRootOfNode::Request request) {

        auto node_root = nodes[request.node].GetRoot();
        return io::FindRootOfNode::Response{node_root};
    }

    io::Response Respond(io::Request request) {
        auto respond = [this](io::Request request) -> io::Response {
            return io::Response{Respond(request)};
        };
        return std::visit(respond, request);
    }
};

Output Solve(const Input &input) {
    Responder responder{input.n_nodes};

    std::vector<io::Response> responses;
    responses.reserve(input.requests.size());

    for (auto request : input.requests) {
        responses.emplace_back(responder.Respond(request));
    }

    return Output{std::move(responses)};
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
    Input input;
    std::optional<Output> optional_expected_output;

    explicit TestIo(Input input) : input{std::move(input)} {
    }

    TestIo(Input input, Output output)
        : input{std::move(input)}, optional_expected_output{std::move(output)} {
    }
};

Output BruteForceSolve(const Input &input) {
    throw NotImplementedError{};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    throw NotImplementedError{};
    Input input;
    return TestIo{input};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    throw NotImplementedError{};
    Input input;
    return TestIo{input};
}

class TimedChecker {
public:
    std::vector<int64_t> durations;

    void Check(const std::string &test_case, const std::string &expected) {
        std::stringstream input_stream{test_case};
        Input input{input_stream};
        Output expected_output{expected};
        TestIo test_io{input, expected_output};
        Check(test_io);
    }

    void Check(TestIo test_io) {
        Output output;
        auto solve = [&output, &test_io]() { output = Solve(test_io.input); };

        durations.emplace_back(detail::Timeit(solve));

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
                ss << "\n================================Expected=============================="
                      "==\n"
                   << expected_output
                   << "\n================================Received=============================="
                      "==\n"
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
        "1 1\n"
        "1",
        "1");

    timed_check.Check(
        "1 1\n"
        "1 1",
        "0");

    timed_check.Check(
        "3 6\n"
        "1 2\n"
        "2 3\n"
        "3 1\n"
        "1\n"
        "2\n"
        "3",
        "1 1 0 1 1 1");

    timed_check.Check(
        "4 22\n"
        "1\n"
        "2\n"
        "3\n"
        "4\n"
        "1 2\n"
        "3 2\n"
        "1\n"
        "2\n"
        "2 3\n"
        "1\n"
        "2\n"
        "3\n"
        "3 4\n"
        "1\n"
        "2\n"
        "3\n"
        "4\n"
        "4 1\n"
        "1\n"
        "2\n"
        "3\n"
        "4\n",
        "1 2 3 4 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1");

    std::cerr << timed_check << "Basic tests OK\n";

    int32_t n_random_test_cases = 100;

    try {

        for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
            timed_check.Check(GenerateRandomTestIo(test_case_id));
        }

        std::cerr << timed_check << "Random tests OK\n";
    } catch (const NotImplementedError &e) {
    }

    int32_t n_stress_test_cases = 1;

    try {
        for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
            timed_check.Check(GenerateStressTestIo(test_case_id));
        }

        std::cerr << timed_check << "Stress tests tests OK\n";
    } catch (const NotImplementedError &e) {
    }

    std::cerr << "OK\n";
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
        std::cout << Solve(Input{std::cin});
    }
    return 0;
}
