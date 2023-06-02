/*
 * Интерфейс прокомментирован с целью объяснить,
 * почему он написан так, а не иначе. В реальной жизни
 * так никто никогда не делает. Комментарии к коду,
 * которые остались бы в его рабочем варианте, заданы
 * с помощью команды однострочного комментария // и написаны
 * на английском языке, как рекомендуется.
 * Остальные комментарии здесь исключительно в учебных целях.
 */

#include <algorithm>
#include <cstring>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

/*
 * Часто в c++ приходится иметь дело с парой итераторов,
 * которые представляют из себя полуинтервал. Например,
 * функция std:sort принимает пару итераторов, участок
 * между которыми нужно отсортировать. В с++11 появился
 * удобный range-based for, который позволяет итерироваться
 * по объекту, для которого определены функции std::begin
 * и std::end (например, это объекты: массив фиксированного
 * размера, любой объект, у которого определены методы
 * begin() и end()). То есть удобный способ итерироваться
 * по std::vector такой:
 * for (const std::string& string: words).
 * Однако, для некоторых объектов существует не один способ
 * итерироваться по ним. Например std::map: мы можем
 * итерироваться по парам объект-значение (как это сделает
 * for(...: map)), или мы можем итерироваться по ключам.
 * Для этого мы можем сделать функцию:
 * IteratorRange<...> keys(const std::map& map),
 * которой можно удобно воспользоваться:
 * for(const std::string& key: keys(dictionary)).
 */
template <class Iterator>
class IteratorRange {
public:
    IteratorRange(Iterator begin, Iterator end) : begin_(begin), end_(end) {
    }

    Iterator begin() const {
        return begin_;
    }

    Iterator end() const {
        return end_;
    }

private:
    Iterator begin_, end_;
};

namespace traverses {

// Traverses the connected component in a breadth-first order
// from the vertex 'origin_vertex'.
// Refer to
// https://goo.gl/0qYXzC
// for the visitor events.
template <class Vertex, class Graph, class Visitor>
void BreadthFirstSearch(Vertex origin_vertex, const Graph &graph, Visitor visitor) {

    visitor.DiscoverVertex(origin_vertex);
    std::unordered_set<Vertex> discovered_vertices{{origin_vertex}};
    std::queue<Vertex> queue{{origin_vertex}};

    while (!queue.empty()) {
        auto vertex = queue.front();
        queue.pop();
        visitor.ExamineVertex(vertex);

        for (auto edge : OutgoingEdges(graph, vertex)) {
            visitor.ExamineEdge(edge);
            Vertex target = GetTarget(graph, edge);

            if (discovered_vertices.count(target) == 0) {
                visitor.DiscoverVertex(target);
                discovered_vertices.insert(target);
                queue.push(target);
            }
        }
    }
}

/*
 * Для начала мы рекомендуем ознакомиться с общей
 * концепцией паттерна проектирования Visitor:
 * https://goo.gl/oZGiYl
 * Для применения Visitor'а к задаче обхода графа
 * можно ознакомиться с
 * https://goo.gl/5gjef2
 */
// See "Visitor Event Points" on
// https://goo.gl/wtAl0y
template <class Vertex, class Edge>
class BfsVisitor {
public:
    virtual void DiscoverVertex(Vertex /*vertex*/) {
    }
    virtual void ExamineEdge(const Edge & /*edge*/) {
    }
    virtual void ExamineVertex(Vertex /*vertex*/) {
    }
    virtual ~BfsVisitor() = default;
};

}  // namespace traverses

namespace aho_corasick {

struct AutomatonNode {
    AutomatonNode() : suffix_link(nullptr), terminal_link(nullptr) {
    }

    // Stores ids of strings which are ended at this node.
    std::vector<size_t> terminated_string_ids;
    // Stores tree structure of nodes.
    std::map<char, AutomatonNode> trie_transitions;
    /*
     * Обратите внимание, что std::set/std::map/std::list
     * при вставке и удалении неинвалидируют ссылки на
     * остальные элементы контейнера. Но стандартные контейнеры
     * std::vector/std::string/std::deque таких гарантий не
     * дают, поэтому хранение указателей на элементы этих
     * контейнеров крайне не рекомендуется.
     */
    // Stores cached transitions of the automaton, contains
    // only pointers to the elements of trie_transitions.
    std::map<char, AutomatonNode *> automaton_transitions_cache;
    AutomatonNode *suffix_link;
    AutomatonNode *terminal_link;
};

// Returns a corresponding trie transition 'nullptr' otherwise.
AutomatonNode *GetTrieTransition(AutomatonNode *node, char character) {
    auto search = node->trie_transitions.find(character);
    if (search != node->trie_transitions.end()) {
        return &search->second;
    }
    return nullptr;
}

// Returns an automaton transition, updates 'node->automaton_transitions_cache'
// if necessary.
// Provides constant amortized runtime.
AutomatonNode *GetAutomatonTransition(AutomatonNode *node, const AutomatonNode *root,
                                      char character) {
    auto cache_search = node->automaton_transitions_cache.find(character);
    if (cache_search != node->automaton_transitions_cache.end()) {
        return cache_search->second;
    }

    if (auto transition = GetTrieTransition(node, character)) {
        return node->automaton_transitions_cache[character] = transition;
    }

    auto original_node = node;
    while (node != root) {
        node = node->suffix_link;
        if (auto transition = GetTrieTransition(node, character)) {
            return original_node->automaton_transitions_cache[character] = transition;
        }
    }

    return original_node->automaton_transitions_cache[character] = node;
}

namespace internal {

class AutomatonGraph {
public:
    struct Edge {
        Edge(AutomatonNode *source, AutomatonNode *target, char character)
            : source(source), target(target), character(character) {
        }

        AutomatonNode *source;
        AutomatonNode *target;
        char character;
    };
};

std::vector<typename AutomatonGraph::Edge> OutgoingEdges(const AutomatonGraph & /*graph*/,
                                                         AutomatonNode *vertex) {
    std::vector<typename AutomatonGraph::Edge> outgoing_edges;
    outgoing_edges.reserve(vertex->trie_transitions.size());
    for (auto &[ch, adjacent_node] : vertex->trie_transitions) {
        outgoing_edges.emplace_back(vertex, &adjacent_node, ch);
    }
    return outgoing_edges;
}

AutomatonNode *GetTarget(const AutomatonGraph & /*graph*/, const AutomatonGraph::Edge &edge) {
    return edge.target;
}

class SuffixLinkCalculator : public traverses::BfsVisitor<AutomatonNode *, AutomatonGraph::Edge> {
public:
    explicit SuffixLinkCalculator(AutomatonNode *root) : root_(root) {
    }

    void ExamineVertex(AutomatonNode *node) override {
        if (not node->suffix_link) {
            node->suffix_link = root_;
        }
    }

    void ExamineEdge(const AutomatonGraph::Edge &edge) override {
        if (edge.source != root_) {
            edge.target->suffix_link =
                GetAutomatonTransition(edge.source->suffix_link, root_, edge.character);
        }
    }

private:
    AutomatonNode *root_;
};

class TerminalLinkCalculator : public traverses::BfsVisitor<AutomatonNode *, AutomatonGraph::Edge> {
public:
    explicit TerminalLinkCalculator(AutomatonNode *root) : root_(root) {
    }

    /*
     * Если вы не знакомы с ключевым словом override,
     * то ознакомьтесь
     * https://goo.gl/u024X0
     */
    void DiscoverVertex(AutomatonNode *node) override {
        if (node == root_ || node->suffix_link->terminated_string_ids.empty()) {
            return;
        }
        node->terminal_link = node->suffix_link;
        node->terminated_string_ids.insert(node->terminated_string_ids.end(),
                                           node->suffix_link->terminated_string_ids.begin(),
                                           node->suffix_link->terminated_string_ids.end());
    }

private:
    AutomatonNode *root_;
};

}  // namespace internal

/*
 * Объясним задачу, которую решает класс NodeReference.
 * Класс Automaton представляет из себя неизменяемый объект
 * (https://goo.gl/4rSP4f),
 * в данном случае, это означает, что единственное действие,
 * которое пользователь может совершать с готовым автоматом,
 * это обходить его разными способами. Это значит, что мы
 * должны предоставить пользователю способ получить вершину
 * автомата и дать возможность переходить между вершинами.
 * Одним из способов это сделать -- это предоставить
 * пользователю константный указатель на AutomatonNode,
 * а вместе с ним константый интерфейс AutomatonNode. Однако,
 * этот вариант ведет к некоторым проблемам.
 * Во-первых, этот же интерфейс AutomatonNode мы должны
 * использовать и для общения автомата с этим внутренним
 * представлением вершины. Так как константная версия
 * этого же интерфейса будет доступна пользователю, то мы
 * ограничены в добавлении функций в этот константный
 * интерфейс (не все функции, которые должны быть доступны
 * автомату должны быть доступны пользователю). Во-вторых,
 * так как мы используем кэширование при переходе по символу
 * в автомате, то условная функция getNextNode не может быть
 * константной (она заполняет кэш переходов). Это значит,
 * что мы лишены возможности добавить функцию "перехода
 * между вершинами" в константный интерфейс (то есть,
 * предоставить ее пользователю константного указателя на
 * AutomatonNode).
 * Во избежание этих проблем, мы создаем отдельный
 * класс, отвечающий ссылке на вершину, который предоставляет
 * пользователю только нужный интерфейс.
 */
class NodeReference {
public:
    NodeReference() : node_(nullptr), root_(nullptr) {
    }

    NodeReference(AutomatonNode *node, AutomatonNode *root) : node_(node), root_(root) {
    }

    NodeReference Next(char character) const {
        return {GetAutomatonTransition(node_, root_, character), root_};
    }

    /*
     * В этом случае есть два хороших способа получить
     * результат работы этой функции:
     * добавить параметр типа OutputIterator, который
     * последовательно записывает в него id найденных
     * строк, или же добавить параметр типа Callback,
     * который будет вызван для каждого такого id.
     * Чтобы ознакомиться с этими концепциями лучше,
     * смотрите ссылки:
     * https://goo.gl/2Kg8wE
     * https://goo.gl/OaUB4k
     * По своей мощности эти способы эквивалентны. (см.
     * https://goo.gl/UaQpPq)
     * Так как в интерфейсе WildcardMatcher мы принимаем
     * Callback, то чтобы пользоваться одним и тем же средством
     * во всех интерфейсах, мы и здесь применяем Callback. Отметим,
     * что другие способы, как например, вернуть std::vector с
     * найденными id, не соответствуют той же степени гибкости, как
     * 2 предыдущие решения (чтобы в этом убедиться представьте
     * себе, как можно решить такую задачу: создать std::set
     * из найденных id).
     */
    template <class Callback>
    void GenerateMatches(Callback on_match) const {
        for (auto id : TerminatedStringIds()) {
            on_match(id);
        }
    }

    bool IsTerminal() const {
        return not node_->terminated_string_ids.empty();
    }

    explicit operator bool() const {
        return node_ != nullptr;
    }

    bool operator==(NodeReference other) const {
        return node_ == other.node_ and root_ == other.root_;
    }

private:
    using TerminatedStringIterator = std::vector<size_t>::const_iterator;
    using TerminatedStringIteratorRange = IteratorRange<TerminatedStringIterator>;

    NodeReference TerminalLink() const {
        return {node_->terminal_link, root_};
    }

    TerminatedStringIteratorRange TerminatedStringIds() const {
        return {node_->terminated_string_ids.begin(), node_->terminated_string_ids.end()};
    }

    AutomatonNode *node_;
    AutomatonNode *root_;
};

class AutomatonBuilder;

class Automaton {
public:
    /*
     * Чтобы ознакомиться с конструкцией =default, смотрите
     * https://goo.gl/jixjHU
     */
    Automaton() = default;

    Automaton(const Automaton &) = delete;
    Automaton &operator=(const Automaton &) = delete;

    NodeReference Root() {
        return {&root_, &root_};
    }

private:
    AutomatonNode root_;

    friend class AutomatonBuilder;
};

class AutomatonBuilder {
public:
    void Add(const std::string &string, size_t id) {
        words_.push_back(string);
        ids_.push_back(id);
    }

    std::unique_ptr<Automaton> Build() {
        auto automaton = std::make_unique<Automaton>();
        BuildTrie(words_, ids_, automaton.get());
        BuildSuffixLinks(automaton.get());
        BuildTerminalLinks(automaton.get());
        return automaton;
    }

private:
    static void BuildTrie(const std::vector<std::string> &words, const std::vector<size_t> &ids,
                          Automaton *automaton) {
        for (size_t i = 0; i < words.size(); ++i) {
            AddString(&automaton->root_, ids[i], words[i]);
        }
    }

    static void AddString(AutomatonNode *root, size_t string_id, const std::string &string) {
        auto node = root;
        for (auto character : string) {
            node = &node->trie_transitions[character];
        }
        node->terminated_string_ids.push_back(string_id);
    }

    static void BuildSuffixLinks(Automaton *automaton) {
        internal::AutomatonGraph graph;
        internal::SuffixLinkCalculator suffix_link_calculator{&automaton->root_};
        traverses::BreadthFirstSearch(&automaton->root_, graph, suffix_link_calculator);
    }

    static void BuildTerminalLinks(Automaton *automaton) {
        internal::AutomatonGraph graph;
        internal::TerminalLinkCalculator terminal_link_calculator{&automaton->root_};
        traverses::BreadthFirstSearch(&automaton->root_, graph, terminal_link_calculator);
    }

    std::vector<std::string> words_;
    std::vector<size_t> ids_;
};

}  // namespace aho_corasick

// Consecutive delimiters are not grouped together and are deemed
// to delimit empty strings
template <class Predicate>
std::vector<std::string> Split(const std::string &string, Predicate is_delimiter) {
    std::vector<std::string> substrings;
    auto previous = string.begin();

    for (auto iter = string.begin(); iter != string.end(); ++iter) {
        if (is_delimiter(*iter)) {
            substrings.emplace_back(previous, iter);
            previous = iter + 1;
        }
    }
    substrings.emplace_back(previous, string.end());

    return substrings;
}

// Wildcard is a character that may be substituted
// for any of all possible characters.
class WildcardMatcher {
public:
    WildcardMatcher() = default;

    explicit WildcardMatcher(const std::vector<std::string> &substrings)
        : number_of_words_{substrings.size()} {
        aho_corasick::AutomatonBuilder builder;
        size_t substring_end = 0;
        for (const auto &substring : substrings) {
            substring_end += substring.size();
            builder.Add(substring, substring_end++);
        }
        words_occurrences_by_position_.resize(substring_end);
        aho_corasick_automaton_ = builder.Build();
        Reset();
    }

    WildcardMatcher static BuildFor(const std::string &pattern, char wildcard) {
        auto substrings = Split(pattern, [=](char character) { return character == wildcard; });
        return WildcardMatcher{substrings};
    }

    // Resets the matcher. Call allows to abandon all data which was already
    // scanned,
    // a new stream can be scanned afterwards.
    void Reset() {
        words_occurrences_by_position_.assign(words_occurrences_by_position_.size(), 0);
        state_ = aho_corasick_automaton_->Root();
        UpdateWordOccurrencesCounters();
    }

    /* В данном случае Callback -- это функция,
     * которая будет вызвана при наступлении
     * события "суффикс совпал с шаблоном".
     * Почему мы выбрали именно этот способ сообщить
     * об этом событии? Можно рассмотреть альтернативы:
     * вернуть bool из Scan, принять итератор и записать
     * в него значение. В первом случае, значение bool,
     * возвращенное из Scan, будет иметь непонятный
     * смысл. True -- в смысле все считалось успешно?
     * True -- произошло совпадение? В случае итератора,
     * совершенно не ясно, какое значение туда  записывать
     * (подошедший суффикс, true, ...?). Более того, обычно,
     * если при сканировании потока мы наткнулись на
     * совпадение, то нам нужно как-то на это отреагировать,
     * это действие и есть наш Callback on_match.
     */
    template <class Callback>
    void Scan(char character, Callback on_match) {
        state_ = state_.Next(character);
        ShiftWordOccurrencesCounters();
        UpdateWordOccurrencesCounters();
        if (words_occurrences_by_position_.back() == number_of_words_) {
            on_match();
        }
    }

private:
    void UpdateWordOccurrencesCounters() {
        auto &words = words_occurrences_by_position_;
        state_.GenerateMatches([&words](size_t id) { ++words[id]; });
    }

    void ShiftWordOccurrencesCounters() {
        words_occurrences_by_position_.pop_back();
        words_occurrences_by_position_.push_front(0);
    }

    // Storing only O(|pattern_with_wildcards|) elements allows us
    // to consume only O(|pattern_with_wildcards|) memory for matcher.
    std::deque<size_t> words_occurrences_by_position_;
    aho_corasick::NodeReference state_;
    size_t number_of_words_ = 0;
    std::unique_ptr<aho_corasick::Automaton> aho_corasick_automaton_;
};

std::string ReadString(std::istream &input_stream) {
    std::string string;
    input_stream >> string;
    return string;
}

// Returns positions of the first character of an every match.
std::vector<size_t> FindFuzzyMatches(const std::string &pattern_with_wildcards,
                                     const std::string &text, char wildcard) {
    auto matcher = WildcardMatcher::BuildFor(pattern_with_wildcards, wildcard);

    std::vector<size_t> occurrences;
    occurrences.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        auto occurrence = i + 1 - pattern_with_wildcards.size();
        matcher.Scan(text[i], [&occurrences, occurrence]() { occurrences.push_back(occurrence); });
    }

    return occurrences;
}

void Print(const std::vector<size_t> &sequence) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cout << sequence.size() << '\n';
    for (auto element : sequence) {
        std::cout << element << ' ';
    }
    std::cout << std::endl;
}
