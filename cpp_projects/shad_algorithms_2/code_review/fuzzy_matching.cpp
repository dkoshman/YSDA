#include "fuzzy_matching.h"

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <utility>
#include <set>

void TestAll();

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--test") == 0) {
        TestAll();
        return 0;
    }
    const char wildcard = '?';
    const std::string pattern_with_wildcards = ReadString(std::cin);
    const std::string text = ReadString(std::cin);
    Print(FindFuzzyMatches(pattern_with_wildcards, text, wildcard));
    return 0;
}

// ===== TESTING ZONE =====

template<class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vector) {
    std::copy(vector.begin(), vector.end(), std::ostream_iterator<T>(os, " "));
    return os;
}

class TestNotPassedException : public std::runtime_error {
public:
    explicit TestNotPassedException(const char *what)
            : std::runtime_error(what) {}

    explicit TestNotPassedException(const std::string &what)
            : std::runtime_error(what.c_str()) {}
};

#define REQUIRE_EQUAL(first, second)                                               \
    do {                                                                           \
        auto firstValue = (first);                                                 \
        auto secondValue = (second);                                               \
        if (!(firstValue == secondValue)) {                                        \
            std::ostringstream oss;                                                \
            oss << "Require equal failed: " << #first << " != " << #second << " (" \
                    << firstValue << " != " << secondValue << ")\n";               \
            throw TestNotPassedException(oss.str());                               \
        }                                                                          \
    } while (false)

void TestBfsHalt();
void TestBfsOrder();
void TestSplit();

void TestAll() {
    {
        aho_corasick::AutomatonBuilder builder;
        builder.Add("suffix", 1);
        builder.Add("ffix", 2);
        builder.Add("ix", 3);
        builder.Add("abba", 4);

        std::unique_ptr<aho_corasick::Automaton> automaton = builder.Build();

        const std::string text = "let us find some suffix";
        aho_corasick::NodeReference node = automaton->Root();
        for (char ch : text) {
            node = node.Next(ch);
        }
        std::vector<size_t> string_ids;

        node.GenerateMatches(
                [&string_ids](size_t string_id) { string_ids.push_back(string_id); });
        std::sort(string_ids.begin(), string_ids.end());

        REQUIRE_EQUAL(string_ids, std::vector<size_t>({1, 2, 3}));
    }
    {
        aho_corasick::AutomatonBuilder builder;
        builder.Add("", 1);
        builder.Add("", 2);
        builder.Add("t", 3);
        builder.Add("", 4);

        auto automaton = builder.Build();
        const std::string text = "test";
        aho_corasick::NodeReference node = automaton->Root();

        std::vector<size_t> sizes{4, 3, 3, 4};
        for (size_t i = 0; i < text.size(); ++i) {
            node = node.Next(text[i]);
            std::set<int> matches;
            node.GenerateMatches([&matches](size_t id) {
                matches.insert(id);
            });
            REQUIRE_EQUAL(sizes[i], matches.size());
        }
    }
    {
        WildcardMatcher matcher = WildcardMatcher::BuildFor("a?c?", '?');

        {
            std::vector<size_t> occurrences;
            //                              012345678901234
            const std::string first_text = "abcaaccxaxcxacc";
            for (size_t i = 0; i < first_text.size(); ++i) {
                matcher.Scan(first_text[i],
                             [&occurrences, i]() { occurrences.push_back(i); });
            }

            REQUIRE_EQUAL(occurrences, std::vector<size_t>({3, 6, 7, 11}));
        }
        {
            matcher.Reset();
            std::vector<size_t> occurrences;
            const std::string second_text = "xyzadcc";
            for (size_t i = 0; i < second_text.size(); ++i) {
                matcher.Scan(second_text[i],
                             [&occurrences, i]() { occurrences.push_back(i); });
            }

            REQUIRE_EQUAL(occurrences, std::vector<size_t>({6}));
        }
    }
    TestSplit();
    TestBfsHalt();
    TestBfsOrder();
    std::cerr << "Tests are passed!\n";
}

void TestSplit() {
    {
        auto is_dash = [](char ch) { return ch == '-'; };
        {
            REQUIRE_EQUAL(Split("a--b-cd-e", is_dash),
                          std::vector<std::string>({"a", "", "b", "cd", "e"}));
        }
        { REQUIRE_EQUAL(Split("-", is_dash), std::vector<std::string>({"", ""})); }
        {
            REQUIRE_EQUAL(Split("--abc--", is_dash),
                          std::vector<std::string>({"", "", "abc", "", ""}));
        }
        { REQUIRE_EQUAL(Split("ab", is_dash), std::vector<std::string>({"ab"})); }
        {
            REQUIRE_EQUAL(Split("", is_dash), std::vector<std::string>({""}));
        }
    }
    {
        auto True = [](char /*ch*/) { return true; };
        {
            std::string s = "2f1tgyhnjd";
            // empty string before each character and one after the string
            REQUIRE_EQUAL(Split(s, True).size(), s.size() + 1);
        }
    }
}

namespace test_bfs {

    typedef std::vector<int> VerticesList;

    struct Graph {
        std::vector<VerticesList> adjacent_vertices;
    };

    int GetTarget(const Graph & /*graph*/, int edge) { return edge; }

    IteratorRange<VerticesList::const_iterator> OutgoingEdges(const Graph &graph,
                                                              int vertex) {
        return IteratorRange<VerticesList::const_iterator>(
                graph.adjacent_vertices[vertex].begin(),
                graph.adjacent_vertices[vertex].end());
    }

    enum class BfsEvent {
        kDiscoverVertex, kExamineEdge, kExamineVertex
    };

    std::ostream& operator<<(std::ostream& out, const BfsEvent& event) {
        switch (event) {
            case BfsEvent::kDiscoverVertex:
                out << "discover";
                break;
            case BfsEvent::kExamineEdge:
                out << "examine edge";
                break;
            case BfsEvent::kExamineVertex:
                out << "examine vertex";
                break;
        }
        return out;
    }

    std::ostream& operator<<(std::ostream& os, const std::pair<BfsEvent, int>& item) {
        os << "(" << item.first << ", " << item.second << ")";
        return os;
    }

    class TestBfsVisitor: public traverses::BfsVisitor<int, int> {
    public:
        explicit TestBfsVisitor(std::vector<std::pair<BfsEvent, int>> *events) {
            events_ = events;
        }

        virtual void DiscoverVertex(int vertex) override {
            events_->emplace_back(BfsEvent::kDiscoverVertex, vertex);
        }
        virtual void ExamineEdge(const int& edge) override {
            events_->emplace_back(BfsEvent::kExamineEdge, edge);
        }
        virtual void ExamineVertex(int vertex) override {
            events_->emplace_back(BfsEvent::kExamineVertex, vertex);
        }

    private:
        std::vector<std::pair<BfsEvent, int>> *events_;
    };
}  // namespace test_bfs

void TestBfsHalt() {
    test_bfs::Graph graph;
    graph.adjacent_vertices.resize(4);
    const int a = 0;
    const int b = 1;
    const int c = 2;
    const int d = 3;
    graph.adjacent_vertices[a].push_back(b);
    graph.adjacent_vertices[b].push_back(c);
    graph.adjacent_vertices[c].push_back(d);
    graph.adjacent_vertices[d].push_back(b);
    traverses::BreadthFirstSearch(a, graph, traverses::BfsVisitor<int, int>());
    // BreadthFirstSearch should not hang on a graph with a cycle
}

void TestBfsOrder() {
    using namespace test_bfs;
    Graph graph;
    graph.adjacent_vertices.resize(4);
    const int a = 0;
    const int b = 1;
    const int c = 2;
    const int d = 3;
    graph.adjacent_vertices[a].push_back(b);
    graph.adjacent_vertices[a].push_back(c);
    graph.adjacent_vertices[b].push_back(d);
    graph.adjacent_vertices[c].push_back(d);
    graph.adjacent_vertices[d].push_back(a);

    std::vector<std::pair<BfsEvent, int>> events;
    TestBfsVisitor visitor(&events);
    traverses::BreadthFirstSearch(a, graph, visitor);

    std::vector<std::pair<BfsEvent, int>> expected{
            {BfsEvent::kDiscoverVertex, a},
            {BfsEvent::kExamineVertex, a},
            {BfsEvent::kExamineEdge, b},
            {BfsEvent::kDiscoverVertex, b},
            {BfsEvent::kExamineEdge, c},
            {BfsEvent::kDiscoverVertex, c},
            {BfsEvent::kExamineVertex, b},
            {BfsEvent::kExamineEdge, d},
            {BfsEvent::kDiscoverVertex, d},
            {BfsEvent::kExamineVertex, c},
            {BfsEvent::kExamineEdge, d},
            {BfsEvent::kExamineVertex, d},
            {BfsEvent::kExamineEdge, a}
    };

    REQUIRE_EQUAL(events, expected);
}