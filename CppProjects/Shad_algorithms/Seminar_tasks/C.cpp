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

struct ColoredRectangle {
    int32_t lower_left_x = 0;
    int32_t lower_left_y = 0;
    int32_t upper_right_x = 0;
    int32_t upper_right_y = 0;
    int32_t color = 0;

    explicit ColoredRectangle(int32_t color) : color{color} {
    }

    explicit ColoredRectangle(std::istream &in) {
        in >> lower_left_x >> lower_left_y >> upper_right_x >> upper_right_y >> color;
    }

    ColoredRectangle(int32_t lower_left_x, int32_t lower_left_y, int32_t upper_right_x,
                     int32_t upper_right_y, int32_t color)
        : lower_left_x{lower_left_x},
          lower_left_y{lower_left_y},
          upper_right_x{upper_right_x},
          upper_right_y{upper_right_y},
          color{color} {
    }
};

class Input {
public:
    ColoredRectangle base_rectangle{1};
    std::vector<ColoredRectangle> stack_of_colored_rectangles;

    Input() = default;

    explicit Input(std::istream &in) {
        int32_t n_rectangles = 0;
        int32_t width = 0;
        int32_t height = 0;
        in >> width >> height >> n_rectangles;
        stack_of_colored_rectangles.reserve(n_rectangles);
        base_rectangle.upper_right_x = width;
        base_rectangle.upper_right_y = height;

        for (int32_t order = 0; order < n_rectangles; ++order) {
            stack_of_colored_rectangles.emplace_back(in);
        }
    }
};

struct ColorArea {
    ColorArea(int32_t color, int32_t area) : color{color}, area{area} {
    }
    int32_t color = 0;
    int32_t area = 0;

    bool operator==(const ColorArea &other) const {
        return color == other.color and area == other.area;
    }
};

class Output {
public:
    std::vector<ColorArea> color_areas;

    Output() = default;

    explicit Output(const std::string &string) {
        std::stringstream ss{string};
        int32_t color = 0;
        int32_t area = 0;
        while (ss >> color >> area) {
            color_areas.emplace_back(color, area);
        }
    }

    std::ostream &Write(std::ostream &out) const {
        for (auto color_area : color_areas) {
            out << color_area.color << ' ' << color_area.area << '\n';
        }
        return out;
    }

    bool operator!=(const Output &other) const {
        return color_areas != other.color_areas;
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

struct RectangleCorner {
    enum class CornerPosition { LowerLeft, UpperRight };

    CornerPosition corner_position;
    int32_t xx = 0;
    int32_t yy = 0;
    int32_t color = 0;
    int32_t order = 0;

    RectangleCorner(CornerPosition corner_position, const io::ColoredRectangle &rectangle,
                    int32_t order)
        : corner_position{corner_position},
          xx{IsLowerLeft() ? rectangle.lower_left_x : rectangle.upper_right_x},
          yy{IsLowerLeft() ? rectangle.lower_left_y : rectangle.upper_right_y},
          color{rectangle.color},
          order{order} {
    }

    [[nodiscard]] bool IsLowerLeft() const {
        return corner_position == CornerPosition::LowerLeft;
    }
};

std::array<RectangleCorner, 2> GetAllCorners(const io::ColoredRectangle &rectangle, int32_t order) {
    return {{{RectangleCorner::CornerPosition::LowerLeft, rectangle, order},
             {RectangleCorner::CornerPosition::UpperRight, rectangle, order}}};
}

bool ComparatorByX(const RectangleCorner &left, const RectangleCorner &right) {
    return left.xx < right.xx;
}

bool ComparatorByY(const RectangleCorner &left, const RectangleCorner &right) {
    return left.yy < right.yy;
}

struct ComparatorByOrder {
    bool operator()(const RectangleCorner &left, const RectangleCorner &right) const {
        return left.order < right.order;
    }
};

template <class Value>
bool ComparatorByColor(const Value &left, const Value &right) {
    return left.color < right.color;
}

template <class ValueType, class Comparator>
class LazyRemoveMaxHeap {
public:
    [[nodiscard]] bool IsEmpty() {
        PopRemovedValues();
        return heap_.empty();
    }

    void Clear() {
        heap_.clear();
        removed_heap_.clear();
    }

    ValueType Front() {
        PopRemovedValues();
        return heap_.front();
    }

    void Pop() {
        PopRemovedValues();
        PopBack(&heap_);
    }

    void Push(const ValueType &value) {
        PushHeap(&heap_, value);
    }

    void Remove(const ValueType &value) {
        PushHeap(&removed_heap_, value);
        PopRemovedValues();
    }

private:
    Comparator comparator_;
    std::vector<ValueType> heap_;
    std::vector<ValueType> removed_heap_;

    void PushHeap(std::vector<ValueType> *heap, ValueType value) {
        heap->push_back(value);
        std::push_heap(heap->begin(), heap->end(), comparator_);
    }

    void PopBack(std::vector<ValueType> *heap) {
        std::pop_heap(heap->begin(), heap->end(), comparator_);
        heap->pop_back();
    }

    void PopRemovedValues() {
        while (not removed_heap_.empty() and
               not comparator_(removed_heap_.front(), heap_.front()) and
               not comparator_(heap_.front(), removed_heap_.front())) {
            PopBack(&heap_);
            PopBack(&removed_heap_);
        }
    }
};

class ColoredRectanglesScanBand {
public:
    std::vector<int32_t> color_areas;

    ColoredRectanglesScanBand(io::ColoredRectangle base_rectangle,
                              const std::vector<io::ColoredRectangle> &rectangles) {

        corners_by_x_.reserve((rectangles.size() + 1) * 2);
        int32_t order = 0;

        for (auto corner_position : GetAllCorners(base_rectangle, order)) {
            corners_by_x_.emplace_back(corner_position);
        }

        for (auto rectangle : rectangles) {
            ++order;
            for (auto corner_position : GetAllCorners(rectangle, order)) {
                corners_by_x_.emplace_back(corner_position);
            }
        }

        std::sort(corners_by_x_.begin(), corners_by_x_.end(), ComparatorByX);

        corners_by_y_ = corners_by_x_;
        std::sort(corners_by_y_.begin(), corners_by_y_.end(), ComparatorByY);

        entered_.resize(order + 1);
        exited_.resize(order + 1);

        auto max_color = std::max_element(rectangles.begin(), rectangles.end(),
                                          ComparatorByColor<io::ColoredRectangle>)
                             ->color;
        color_areas.resize(max_color + 1);
    }

    void Scan() {
        auto previous_scanned_x = corners_by_x_.front().xx;

        for (auto corner : corners_by_x_) {

            if (previous_scanned_x < corner.xx) {
                ScanLine(corner.xx - previous_scanned_x);
                previous_scanned_x = corner.xx;
            }

            if (corner.IsLowerLeft()) {
                entered_[corner.order] = true;
            } else {
                exited_[corner.order] = true;
            }
        }
    }

private:
    std::vector<RectangleCorner> corners_by_x_;
    std::vector<RectangleCorner> corners_by_y_;
    std::vector<bool> entered_;
    std::vector<bool> exited_;
    LazyRemoveMaxHeap<RectangleCorner, ComparatorByOrder> max_order_heap_;

    void ScanLine(int32_t current_scan_band_width) {

        max_order_heap_.Clear();
        int32_t previous_scanned_y = corners_by_y_.front().yy;

        for (auto corner : corners_by_y_) {
            if (not entered_[corner.order] or exited_[corner.order]) {
                continue;
            }

            if (previous_scanned_y < corner.yy) {
                auto current_scan_band_height = corner.yy - previous_scanned_y;
                color_areas[max_order_heap_.Front().color] +=
                    current_scan_band_width * current_scan_band_height;
                previous_scanned_y = corner.yy;
            }

            if (corner.IsLowerLeft()) {
                max_order_heap_.Push(corner);
            } else {
                max_order_heap_.Remove(corner);
            }
        }
    }
};

io::Output Solve(const io::Input &input) {
    ColoredRectanglesScanBand scan_line{input.base_rectangle, input.stack_of_colored_rectangles};

    scan_line.Scan();

    io::Output output;
    for (int32_t color = 0; color < static_cast<int32_t>(scan_line.color_areas.size()); ++color) {
        if (scan_line.color_areas[color] != 0) {
            output.color_areas.emplace_back(color, scan_line.color_areas[color]);
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
        throw std::invalid_argument{"IsEmpty container."};
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
    throw NotImplementedError{};
}

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_rectangles = 1 + test_case_id;
    int32_t base_rectangle_size = 1 + test_case_id;
    int32_t color_count = 1 + test_case_id;

    std::uniform_int_distribution position_dist(0, base_rectangle_size);
    std::uniform_int_distribution color_dist(1, color_count);
    auto &engine = *rng::GetEngine();

    io::Input input;
    input.base_rectangle.upper_right_y = input.base_rectangle.upper_right_x = base_rectangle_size;
    input.stack_of_colored_rectangles.reserve(n_rectangles * 2);

    for (int32_t order = 0; order < n_rectangles; ++order) {

        auto lower_left_x = position_dist(engine);
        auto upper_right_x = position_dist(engine);
        if (lower_left_x > upper_right_x) {
            std::swap(lower_left_x, upper_right_x);
        }

        auto lower_left_y = position_dist(engine);
        auto upper_right_y = position_dist(engine);
        if (lower_left_y > upper_right_y) {
            std::swap(lower_left_y, upper_right_y);
        }

        auto color = color_dist(engine);

        input.stack_of_colored_rectangles.emplace_back(lower_left_x, lower_left_y, upper_right_x,
                                                       upper_right_y, color);
    }

    std::vector<io::ColorArea> colors;
    colors.reserve(color_count + 1);
    for (int i = 0; i < color_count + 1; ++i) {
        colors.emplace_back(i, 0);
    }

    std::vector<std::vector<int32_t>> board(base_rectangle_size);
    for (auto &line : board) {
        line.resize(base_rectangle_size);
    }

    int32_t overall_area = 0;
    for (auto iterator = input.stack_of_colored_rectangles.rbegin();
         iterator != input.stack_of_colored_rectangles.rend(); ++iterator) {

        auto rectangle = *iterator;
        for (int32_t x = rectangle.lower_left_x; x < rectangle.upper_right_x; ++x) {
            for (int32_t y = rectangle.lower_left_y; y < rectangle.upper_right_y; ++y) {
                if (board[x][y] == 0) {
                    board[x][y] = rectangle.color;
                    ++colors[rectangle.color].area;
                    ++overall_area;
                }
            }
        }
    }
    colors[1].area += base_rectangle_size * base_rectangle_size - overall_area;

    io::Output output;
    output.color_areas.reserve(color_count);
    for (auto color : colors) {
        if (color.area != 0) {
            output.color_areas.push_back(color);
        }
    }

    return TestIo{input, output};
}

TestIo GenerateStressTestIo([[maybe_unused]] int32_t test_case_id) {
    return GenerateRandomTestIo(1000);
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
        "20 19 3\n"
        "2 2 18 18 2\n"
        "0 8 20 19 3\n"
        "8 0 10 19 4\n",
        "1 60\n"
        "2 84\n"
        "3 198\n"
        "4 38\n");

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
