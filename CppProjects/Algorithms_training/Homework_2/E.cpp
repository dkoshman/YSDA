#include <algorithm>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cmath>

struct Coordinate {
    int64_t x, y;
};

int64_t CountCutAndWateredBundles(std::istream& in = std::cin) {
    Coordinate first_corner, second_corner, irrigator;
    for (auto& coordinate_p : {&first_corner, &second_corner, &irrigator}) {
        in >> coordinate_p->x >> coordinate_p->y;
    }
    int64_t irrigator_radius;
    in >> irrigator_radius;
    auto x_range = std::minmax(first_corner.x, second_corner.x);
    auto y_range = std::minmax(first_corner.y, second_corner.y);
    int64_t bundles = 0;
    int64_t radius_squared = std::pow(irrigator_radius, 2);

    for (int64_t x = x_range.first; x <= x_range.second; ++x) {
        if (std::abs(x - irrigator.x) <= irrigator_radius) {
            int64_t y_radius = std::sqrt(radius_squared - std::pow(x - irrigator.x, 2));
            int64_t diff = std::min(y_radius + irrigator.y, y_range.second) -
                           std::max(-y_radius + irrigator.y, y_range.first) + 1;
            if (diff > 0) {
                bundles += diff;
            }
        }
    }
    return bundles;
}

void Test() {
    std::stringstream stream;
    stream << "0 0 5 4 4 0 3";
    assert(CountCutAndWateredBundles(stream) == 14);
    stream.clear();
    stream << "0 0 4 4 0 0 10";
    assert(CountCutAndWateredBundles(stream) == 25);
    stream.clear();
    stream << "0 0 4 4 0 0 1";
    assert(CountCutAndWateredBundles(stream) == 3);
    stream.clear();
    stream << "0 0 4 4 0 0 0";
    assert(CountCutAndWateredBundles(stream) == 1);
    stream.clear();
    stream << "0 0 4 4 2 2 2";
    assert(CountCutAndWateredBundles(stream) == 13);
    stream.clear();
    stream << "0 0 4 4 2 -10000 2";

    assert(CountCutAndWateredBundles(stream) == 0);
    stream.clear();
    stream << "-1 -2 3 0 -2 1 3";
    assert(CountCutAndWateredBundles(stream) == 4);
    stream.clear();
    stream << "-1 -2 0 0 -3 -4 5";
    assert(CountCutAndWateredBundles(stream) == 6);
    stream.clear();
    stream << "-100000 -100000 100000 100000 0 0 100000000";
    //    std::cout << CountCutAndWateredBundles(stream);
    assert(CountCutAndWateredBundles(stream) ==
           static_cast<int64_t>(200001) * static_cast<int64_t>(200001));
}

int main() {
    Test();
    std::cout << CountCutAndWateredBundles() << std::endl;
    return 0;
}
