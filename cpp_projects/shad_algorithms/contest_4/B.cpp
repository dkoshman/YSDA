#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
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

template <class Dividend, class Divisor>
Divisor PositiveMod(Dividend value, Divisor divisor) {
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

enum class DirectionInClockwiseOrder { down, left, up, right };

class Point {
public:
    int32_t xx = 0;
    int32_t yy = 0;

    Point() = default;

    Point(int32_t xx, int32_t yy) : xx{xx}, yy{yy} {
    }

    [[nodiscard]] bool IsInThisDirectionFrom(DirectionInClockwiseOrder direction,
                                             const Point &other) const {
        switch (direction) {
            case DirectionInClockwiseOrder::down:
                return yy < other.yy;
            case DirectionInClockwiseOrder::left:
                return xx < other.xx;
            case DirectionInClockwiseOrder::up:
                return yy > other.yy;
            case DirectionInClockwiseOrder::right:
                return xx > other.xx;
        }
        throw std::invalid_argument("Unknown direction.");
    }
};

std::istream &operator>>(std::istream &in, Point &point) {
    in >> point.xx >> point.yy;
    return in;
}

enum class ExternalRelativePositionOf2RectanglesInClockwiseOrder {
    left_and_not_up,
    up_and_not_right,
    right_and_not_down,
    down_and_not_left,
};

enum class NonExternalRelativePositionOf2Rectangles {
    interior,
    exterior,
    intersecting,
};

using RelativePositionOf2Rectangles =
    std::variant<ExternalRelativePositionOf2RectanglesInClockwiseOrder,
                 NonExternalRelativePositionOf2Rectangles>;

struct RelativeDirectionsOf2Rectangles {
    DirectionInClockwiseOrder main_direction;
    DirectionInClockwiseOrder opposite_secondary_direction;

    RelativeDirectionsOf2Rectangles(DirectionInClockwiseOrder main_direction,
                                    DirectionInClockwiseOrder opposite_secondary_direction)
        : main_direction{main_direction},
          opposite_secondary_direction{opposite_secondary_direction} {
    }

    bool operator==(const RelativeDirectionsOf2Rectangles &other) const {
        return std::tie(main_direction, opposite_secondary_direction) ==
               std::tie(other.main_direction, other.opposite_secondary_direction);
    }
};

RelativeDirectionsOf2Rectangles GetRelativeDirectionsByPosition(
    ExternalRelativePositionOf2RectanglesInClockwiseOrder position) {
    switch (position) {
        case ExternalRelativePositionOf2RectanglesInClockwiseOrder::left_and_not_up:
            return {DirectionInClockwiseOrder::left, DirectionInClockwiseOrder::up};
        case ExternalRelativePositionOf2RectanglesInClockwiseOrder::up_and_not_right:
            return {DirectionInClockwiseOrder::up, DirectionInClockwiseOrder::right};
        case ExternalRelativePositionOf2RectanglesInClockwiseOrder::right_and_not_down:
            return {DirectionInClockwiseOrder::right, DirectionInClockwiseOrder::down};
        case ExternalRelativePositionOf2RectanglesInClockwiseOrder::down_and_not_left:
            return {DirectionInClockwiseOrder::down, DirectionInClockwiseOrder::left};
        default:
            throw std::invalid_argument("Relative position doesn't correspond to directions.");
    }
}

class Rectangle {
public:
    Point lower_left;
    Point upper_right;

    Rectangle() = default;

    Rectangle(int32_t lower_left_x, int32_t lower_left_y, int32_t upper_right_x,
              int32_t upper_right_y)
        : lower_left{lower_left_x, lower_left_y}, upper_right{upper_right_x, upper_right_y} {
        Reset();
    }

    void Reset() {
        if (lower_left.xx > upper_right.xx) {
            std::swap(lower_left.xx, upper_right.xx);
        }
        if (lower_left.yy > upper_right.yy) {
            std::swap(lower_left.yy, upper_right.yy);
        }
    }

    [[nodiscard]] int64_t GetPerimeter() const {
        return 2 * (static_cast<int64_t>(upper_right.xx) + upper_right.yy - lower_left.xx -
                    lower_left.yy);
    }

    [[nodiscard]] bool IsInThisDirectionFrom(DirectionInClockwiseOrder direction,
                                             const Rectangle &other) const {
        return lower_left.IsInThisDirectionFrom(direction, other.lower_left) and
               lower_left.IsInThisDirectionFrom(direction, other.upper_right) and
               upper_right.IsInThisDirectionFrom(direction, other.lower_left) and
               upper_right.IsInThisDirectionFrom(direction, other.upper_right);
    }

    [[nodiscard]] bool IsExteriorTo(const Rectangle &other) const {
        return lower_left.IsInThisDirectionFrom(DirectionInClockwiseOrder::left,
                                                other.lower_left) and
               lower_left.IsInThisDirectionFrom(DirectionInClockwiseOrder::down,
                                                other.lower_left) and
               upper_right.IsInThisDirectionFrom(DirectionInClockwiseOrder::right,
                                                 other.upper_right) and
               upper_right.IsInThisDirectionFrom(DirectionInClockwiseOrder::up, other.upper_right);
    }

    [[nodiscard]] bool IsInTheseRelativeDirectionsFrom(RelativeDirectionsOf2Rectangles directions,
                                                       const Rectangle &other) const {
        return IsInThisDirectionFrom(directions.main_direction, other) and
               not IsInThisDirectionFrom(directions.opposite_secondary_direction, other);
    }

    [[nodiscard]] RelativePositionOf2Rectangles GetRelativePositionTo(
        const Rectangle &other) const {

        if (IsExteriorTo(other)) {
            return NonExternalRelativePositionOf2Rectangles::exterior;
        }

        if (other.IsExteriorTo(*this)) {
            return NonExternalRelativePositionOf2Rectangles::interior;
        }

        auto external_positions = {
            ExternalRelativePositionOf2RectanglesInClockwiseOrder::left_and_not_up,
            ExternalRelativePositionOf2RectanglesInClockwiseOrder::up_and_not_right,
            ExternalRelativePositionOf2RectanglesInClockwiseOrder::right_and_not_down,
            ExternalRelativePositionOf2RectanglesInClockwiseOrder::down_and_not_left};
        for (auto position : external_positions) {
            if (IsInTheseRelativeDirectionsFrom(GetRelativeDirectionsByPosition(position), other)) {
                return position;
            }
        }

        return NonExternalRelativePositionOf2Rectangles::intersecting;
    }
};

std::istream &operator>>(std::istream &in, Rectangle &rectangle) {
    in >> rectangle.lower_left >> rectangle.upper_right;
    rectangle.Reset();
    return in;
}

class ExternalRectangleNode {
public:
    Rectangle rectangle;
    std::array<std::unique_ptr<ExternalRectangleNode>, 4> children;

    explicit ExternalRectangleNode(Rectangle rectangle) : rectangle{rectangle} {
    }

    [[nodiscard]] int32_t GetSubtreeSize() const {
        int32_t size = 1;
        for (auto &child : children) {
            if (child) {
                size += child->GetSubtreeSize();
            }
        }
        return size;
    }

    void Insert(const Rectangle &new_rectangle) {
        if (IsRectangleInteriorToAnyNodes(new_rectangle)) {
            return;
        }

        if (not new_rectangle.IsExteriorTo(this->rectangle)) {
            RemoveChildNodesInteriorToRectangle(new_rectangle);
            InsertExternalRectangle(new_rectangle);
            return;
        }

        auto non_interior_children = CollectAllChildRectanglesNonInteriorTo(new_rectangle);

        this->rectangle = new_rectangle;

        for (auto &child : children) {
            child.reset();
        }

        for (auto exterior_rectangle : non_interior_children) {
            InsertExternalRectangle(exterior_rectangle);
        }
    }

private:
    [[nodiscard]] int32_t GetChildIdFromRelativePosition(
        ExternalRelativePositionOf2RectanglesInClockwiseOrder relative_position) const {
        return static_cast<int32_t>(relative_position);
    }

    [[nodiscard]] int32_t GetChildIdCorrespondingToRectangle(const Rectangle &new_rectangle) const {

        auto new_rectangle_relative_position = new_rectangle.GetRelativePositionTo(this->rectangle);

        if (auto external_position =
                std::get_if<ExternalRelativePositionOf2RectanglesInClockwiseOrder>(
                    &new_rectangle_relative_position)) {
            return GetChildIdFromRelativePosition(*external_position);
        }

        throw std::invalid_argument{"Rectangle is not in external position relative to this node."};
    }

    [[nodiscard]] int32_t PositiveModNChildren(int32_t value) const {
        auto n_children = static_cast<int32_t>(children.size());
        return PositiveMod(value, n_children);
    }

    [[nodiscard]] int32_t GetClockwiseNeighborOfChild(int32_t child_id) const {
        return PositiveModNChildren(child_id + 1);
    }

    [[nodiscard]] int32_t GetCounterClockwiseNeighborOfChild(int32_t child_id) const {
        return PositiveModNChildren(child_id - 1);
    }

    void InsertExternalRectangle(const Rectangle &new_rectangle) {
        auto child_id = GetChildIdCorrespondingToRectangle(new_rectangle);
        auto &child = children[child_id];
        if (child) {
            child->InsertExternalRectangle(new_rectangle);
        } else {
            child = std::make_unique<ExternalRectangleNode>(new_rectangle);
        }
    }

    [[nodiscard]] bool IsRectangleInteriorToAnyNodes(const Rectangle &new_rectangle) const {
        auto new_rectangle_relative_position = new_rectangle.GetRelativePositionTo(this->rectangle);

        if (auto non_external_position = std::get_if<NonExternalRelativePositionOf2Rectangles>(
                &new_rectangle_relative_position)) {
            switch (*non_external_position) {
                case (NonExternalRelativePositionOf2Rectangles::interior):
                    return true;
                case (NonExternalRelativePositionOf2Rectangles::exterior):
                    return false;
                case (NonExternalRelativePositionOf2Rectangles::intersecting):
                    throw std::invalid_argument{"Intersecting rectangles."};
            }
        }

        auto child_id = GetChildIdCorrespondingToRectangle(new_rectangle);

        auto children_to_check_for_rectangles_exterior_to_new_one = {
            child_id, GetCounterClockwiseNeighborOfChild(child_id)};

        for (auto candidate_child_id : children_to_check_for_rectangles_exterior_to_new_one) {
            auto &child = children[candidate_child_id];
            if (child and child->IsRectangleInteriorToAnyNodes(new_rectangle)) {
                return true;
            }
        }

        return false;
    }

    void RemoveChildNodesInteriorToRectangle(const Rectangle &new_rectangle) {
        auto child_id = GetChildIdCorrespondingToRectangle(new_rectangle);

        auto children_to_check_for_rectangles_interior_to_new_one = {
            child_id, GetClockwiseNeighborOfChild(child_id)};

        for (auto candidate_child_id : children_to_check_for_rectangles_interior_to_new_one) {
            auto &child = children[candidate_child_id];

            if (child) {
                if (not new_rectangle.IsExteriorTo(child->rectangle)) {
                    child->RemoveChildNodesInteriorToRectangle(new_rectangle);

                } else {
                    auto non_interior_children_rectangles =
                        child->CollectAllChildRectanglesNonInteriorTo(new_rectangle);

                    child.reset();

                    for (auto exterior_rectangle : non_interior_children_rectangles) {
                        InsertExternalRectangle(exterior_rectangle);
                    }
                }
            }
        }
    }

    std::vector<Rectangle> CollectAllChildRectanglesNonInteriorTo(const Rectangle &new_rectangle) {
        std::vector<Rectangle> non_interior_rectangles;
        RecursiveCollectAllChildRectanglesNonInteriorTo(new_rectangle, &non_interior_rectangles);
        return non_interior_rectangles;
    }

    void RecursiveCollectAllChildRectanglesNonInteriorTo(
        const Rectangle &new_rectangle, std::vector<Rectangle> *non_interior_rectangles) {

        if (not new_rectangle.IsExteriorTo(this->rectangle)) {
            non_interior_rectangles->push_back(this->rectangle);
        }
        for (auto &child : children) {
            if (child) {
                child->RecursiveCollectAllChildRectanglesNonInteriorTo(new_rectangle,
                                                                       non_interior_rectangles);
            }
        }
    }
};

namespace io {

class InputType {
public:
    std::vector<Rectangle> rectangles;

    InputType() = default;

    explicit InputType(std::istream &in) {
        size_t rectangle_count = 0;
        in >> rectangle_count;

        rectangles.resize(rectangle_count);
        for (auto &rectangle : rectangles) {
            in >> rectangle;
        }
    }
};

class OutputType {
public:
    OutputType() = default;

    explicit OutputType(int32_t n_exterior_rectangles)
        : n_exterior_rectangles{n_exterior_rectangles} {
    }

    explicit OutputType(const std::string &string) {
        std::stringstream ss{string};
        ss >> n_exterior_rectangles;
    }

    std::ostream &Write(std::ostream &out) const {
        out << n_exterior_rectangles << '\n';
        return out;
    }

    bool operator!=(const OutputType &other) const {
        return n_exterior_rectangles != other.n_exterior_rectangles;
    }

    int32_t n_exterior_rectangles = 0;
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

}  // namespace io

using io::InputType, io::OutputType;

OutputType Solve(const InputType &input) {
    auto rectangles = input.rectangles;

    std::shuffle(rectangles.begin(), rectangles.end(), *rng::GetEngine());

    ExternalRectangleNode root{rectangles.front()};

    for (size_t i = 1; i < rectangles.size(); ++i) {
        root.Insert(rectangles[i]);
    }

    return OutputType{root.GetSubtreeSize()};
}

namespace test {

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

class ComparatorRectanglesPerimeterGreater {
public:
    bool operator()(const Rectangle &lhv, const Rectangle &rhv) const {
        return lhv.GetPerimeter() > rhv.GetPerimeter();
    }
};

OutputType BruteForceSolve(const InputType &input) {
    auto rectangles = input.rectangles;

    std::sort(rectangles.begin(), rectangles.end(), ComparatorRectanglesPerimeterGreater{});

    std::vector<Rectangle> exterior_rectangles;

    for (auto &rectangle : rectangles) {
        auto is_interior = [rectangle](const Rectangle &other) -> bool {
            return other.IsExteriorTo(rectangle);
        };
        if (not std::any_of(exterior_rectangles.begin(), exterior_rectangles.end(), is_interior)) {
            exterior_rectangles.push_back(rectangle);
        }
    }

    return OutputType{static_cast<int32_t>(exterior_rectangles.size())};
}

std::optional<Rectangle> GenerateRandomRectangleInsideGivenRectangle(
    const Rectangle &exterior_rectangle) {
    auto adjusted_upper_right = exterior_rectangle.upper_right;
    --adjusted_upper_right.xx;
    --adjusted_upper_right.yy;
    if (not(exterior_rectangle.lower_left.xx < adjusted_upper_right.xx and
            exterior_rectangle.lower_left.yy < adjusted_upper_right.yy)) {
        return std::nullopt;
    }

    std::uniform_int_distribution<int32_t> x_distribution{exterior_rectangle.lower_left.xx + 1,
                                                          exterior_rectangle.upper_right.xx - 1};
    std::uniform_int_distribution<int32_t> y_distribution{exterior_rectangle.lower_left.yy + 1,
                                                          exterior_rectangle.upper_right.yy - 1};
    auto &engine = *rng::GetEngine();
    Rectangle rectangle;
    int32_t n_tries = 10;
    do {
        rectangle = Rectangle{x_distribution(engine), y_distribution(engine),
                              x_distribution(engine), y_distribution(engine)};
        n_tries--;
        if (n_tries < 0) {
            return std::nullopt;
        }
    } while (rectangle.lower_left.xx == rectangle.upper_right.xx or
             rectangle.lower_left.yy == rectangle.upper_right.yy);
    assert(exterior_rectangle.IsExteriorTo(rectangle));
    return rectangle;
}

struct TestIo {
    InputType input;
    std::optional<OutputType> optional_expected_output = std::nullopt;

    explicit TestIo(InputType input) {
        try {
            optional_expected_output = BruteForceSolve(input);
        } catch (const NotImplementedError &e) {
        }
        this->input = std::move(input);
    }

    TestIo(InputType input, OutputType output)
        : input{std::move(input)}, optional_expected_output{output} {
    }
};

TestIo GenerateRandomTestIo(int32_t test_case_id) {
    int32_t n_rectangles = 1 + test_case_id * 3;
    int32_t n_exterior_rectangles = 1 + test_case_id;
    int32_t min_coordinate = -2 - test_case_id * 3;
    int32_t max_coordinate = 2 + test_case_id * 3;

    //    int32_t n_rectangles = 5;
    //    int32_t n_exterior_rectangles = 2;
    //    int32_t min_coordinate = -10;
    //    int32_t max_coordinate = 10;

    int32_t root_home_strip_width = (max_coordinate - min_coordinate) / n_exterior_rectangles;
    InputType input;
    std::vector<Rectangle> exterior_rectangles;
    for (int32_t i = 0; i < n_exterior_rectangles; ++i) {
        auto rectangle = GenerateRandomRectangleInsideGivenRectangle(
            {min_coordinate, min_coordinate + i * root_home_strip_width, max_coordinate,
             min_coordinate + (i + 1) * root_home_strip_width});
        assert(rectangle.value().upper_right.yy < min_coordinate + (i + 1) * root_home_strip_width);
        assert(rectangle.value().lower_left.yy > min_coordinate + i * root_home_strip_width);
        exterior_rectangles.push_back(rectangle.value());
    }

    input.rectangles = exterior_rectangles;
    std::uniform_int_distribution<int32_t> distribution{
        0, static_cast<int32_t>(exterior_rectangles.size() - 1)};

    int32_t max_tries = 1'000'000;
    while (input.rectangles.size() < static_cast<size_t>(n_rectangles) and max_tries > 0) {
        --max_tries;
        auto exterior_id = distribution(*rng::GetEngine());
        auto rectangle =
            GenerateRandomRectangleInsideGivenRectangle(exterior_rectangles[exterior_id]);
        if (rectangle) {
            input.rectangles.push_back(rectangle.value());
            exterior_rectangles[exterior_id] = rectangle.value();
        }
    }

    assert(BruteForceSolve(input).n_exterior_rectangles == n_exterior_rectangles);

    return {input, OutputType{n_exterior_rectangles}};
}

TestIo GenerateStressTestIo(int32_t test_case_id) {
    int32_t n_exterior_rectangles = 100'000;
    int32_t min_coordinate = -1'000'000'000;
    int32_t max_coordinate = 1'000'000'000;

    int32_t x_band_width = (max_coordinate - min_coordinate) / n_exterior_rectangles;
    InputType input;
    for (int32_t i = 0; i < n_exterior_rectangles; ++i) {
        input.rectangles.emplace_back(min_coordinate + i * x_band_width, min_coordinate,
                                      min_coordinate + (i + 1) * x_band_width - 1,
                                      min_coordinate + i + 1);
    }
    return {input, OutputType{n_exterior_rectangles}};
}

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

int64_t Check(const TestIo &test_io) {

    TimeItInMilliseconds time;
    auto output = Solve(test_io.input);
    time.End();

    if (test_io.optional_expected_output) {
        auto expected_output = test_io.optional_expected_output.value();

        if (output != expected_output) {
            Solve(test_io.input);
            std::stringstream ss;
            ss << "\n==============================Expected==============================\n"
               << expected_output
               << "\n==============================Received==============================\n"
               << output << "\n";
            throw WrongAnswerException{ss.str()};
        }
    }

    return time.Duration();
}

int64_t Check(const std::string &test_case, int32_t expected) {
    std::stringstream input_stream{test_case};
    return Check({InputType{input_stream}, OutputType{expected}});
}

struct Stats {
    double mean = 0;
    double std = 0;
    double max = 0;
};

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

void Test() {
    rng::PrintSeed();

    Check(
        "3\n"
        "-3 -3 3 3\n"
        "-2 2 2 -2\n"
        "-1 -1 1 1\n",
        1);
    Check(
        "4\n"
        "0 0 3 3\n"
        "1 1 2 2\n"
        "100 100 101 101\n"
        "200 200 201 201\n",
        3);
    Check(
        "1\n"
        "0 0 3 3\n",
        1);
    Check(
        "1\n"
        "-1000000000 -1000000000 1000000000 1000000000\n",
        1);
    Check(
        "2\n"
        "-999999999 -999999999 999999999 999999999\n"
        "-1000000000 -1000000000 1000000000 1000000000\n",
        1);
    Check(
        "2\n"
        "-1000000000 -1000000000 1000000000 0\n"
        "-1000000000 1 1000000000 1000000000\n",
        2);
    Check(
        "2\n"
        "-1000000000 -1000000000 0 1000000000\n"
        "1 -1000000000 1000000000 1000000000\n",
        2);

    std::cerr << "Basic tests OK\n";

    std::vector<int64_t> durations;
    TimeItInMilliseconds time_it;

    int32_t n_random_test_cases = 100;
    for (int32_t test_case_id = 0; test_case_id < n_random_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateRandomTestIo(test_case_id)));
    }

    std::cerr << "Random tests OK\n";

    int32_t n_stress_test_cases = 1;
    for (int32_t test_case_id = 0; test_case_id < n_stress_test_cases; ++test_case_id) {
        durations.emplace_back(Check(GenerateStressTestIo(test_case_id)));
    }

    std::cerr << "Stress tests tests OK\n";

    auto duration_stats = ComputeStats(durations.begin(), durations.end());
    std::cerr << "Solve duration stats in milliseconds:\n"
              << "\tMean:\t" + std::to_string(duration_stats.mean) << '\n'
              << "\tStd:\t" + std::to_string(duration_stats.std) << '\n'
              << "\tMax:\t" + std::to_string(duration_stats.max) << '\n';

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
        std::cout << Solve(InputType{std::cin});
    }
    return 0;
}
