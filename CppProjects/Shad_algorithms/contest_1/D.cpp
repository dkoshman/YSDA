#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

struct Point2d {
    int first = 0;
    int second = 0;
};

class InputType {
public:
    explicit InputType(std::istream &in) {
        in >> n_points >> n_circles_to_cover;
        points.resize(n_points);
        for (auto &[x, y] : points) {
            in >> x >> y;
        }
    }

    std::vector<Point2d> points;
    int32_t n_points = 0;
    int32_t n_circles_to_cover = 0;
};

class OutputType {
public:
    OutputType() = default;

    explicit OutputType(double smallest_covering_circle_radius)
        : smallest_covering_circle_radius{smallest_covering_circle_radius} {
    }

    explicit OutputType(const std::string &string) {
        std::stringstream ss{string};
        ss >> smallest_covering_circle_radius;
    }

    bool operator!=(const OutputType &other) const {

        return smallest_covering_circle_radius + 0.001 < other.smallest_covering_circle_radius or
               other.smallest_covering_circle_radius < smallest_covering_circle_radius - 0.001;
    };

    std::ostream &Write(std::ostream &out) const {
        out << std::fixed << std::setprecision(6) << smallest_covering_circle_radius << '\n';
        return out;
    }

    double smallest_covering_circle_radius = 0;
};

std::ostream &operator<<(std::ostream &os, OutputType const &output) {
    return output.Write(os);
}

class WrongAnswerException : public std::exception {
public:
    explicit WrongAnswerException(std::string const &message) : message{message.data()} {
    }

    const char *what() const noexcept {
        return message;
    }

    const char *message;
};

class AxisIntersectionsWithCircles {
public:
    explicit AxisIntersectionsWithCircles(size_t n_circles)
        : left_intersections(n_circles), right_intersections(n_circles) {
    }
    std::vector<double> left_intersections;
    std::vector<double> right_intersections;

    void RecalculateAxisIntersectionsWithCircles(const std::vector<Point2d> &circle_centers,
                                                 double radius) {
        left_intersections.clear();
        right_intersections.clear();
        double radius_squared = radius * radius;

        for (const auto &[x, y] : circle_centers) {
            if (abs(y) <= radius) {
                double half = std::sqrt(radius_squared - y * y);
                left_intersections.push_back(x - half);
                right_intersections.push_back(x + half);
            }
        }
    }

    int32_t FindMaxNumberOfNestedIntersectionIntervals() {
        std::sort(left_intersections.begin(), left_intersections.end());
        std::sort(right_intersections.begin(), right_intersections.end());
        int32_t max_n_nested_intersections = 0;
        int32_t n_intersections = static_cast<int>(left_intersections.size());
        int32_t n_current_intersections = 0;
        int32_t left_index = 0;
        int32_t right_index = 0;

        while (left_index < n_intersections) {
            if (left_intersections[left_index] < right_intersections[right_index]) {
                ++n_current_intersections;
                max_n_nested_intersections =
                    std::max(max_n_nested_intersections, n_current_intersections);
                ++left_index;
            } else {
                --n_current_intersections;
                ++right_index;
            }
        }
        return max_n_nested_intersections;
    }
};

OutputType Solve(InputType input) {
    double lower = 0;
    double upper = 1500;
    AxisIntersectionsWithCircles intersections(input.n_points);

    while (upper - lower > 0.0001) {
        double middle = (upper + lower) / 2;
        intersections.RecalculateAxisIntersectionsWithCircles(input.points, middle);
        if (intersections.FindMaxNumberOfNestedIntersectionIntervals() < input.n_circles_to_cover) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    return OutputType{(upper + lower) / 2};
}

void Check(const std::string &test_case, const std::string &expected) {
    std::stringstream input_stream{test_case};
    auto input = InputType{input_stream};
    auto output = Solve(input);
    if (output != OutputType{expected}) {
        std::stringstream ss;
        ss << "\nExpected:\n" << expected << "\nReceived:" << output << "\n";
        throw WrongAnswerException{ss.str()};
    }
}

void Test() {
    Check(
        "3 3\n"
        "0 5\n"
        "3 4\n"
        "-4 -3\n",
        "5");
    Check(
        "3 2\n"
        "0 1\n"
        "2 1\n"
        "1 100\n",
        "1.414246");
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
