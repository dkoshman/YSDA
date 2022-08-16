#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <set>

using namespace std;

long long abs(long long n) {
    if (n < 0) {
        n *= (-1);
    }
    return n;
}

struct point {
    long long x;
    long long y;
    int ind;
};

bool operator< (point p1, point p2) {
    if (p1.y == p2.y) {
        return p1.x < p2.x;
    }

    return p1.y < p2.y;
}

int main() {
    int n;
    cin >> n;
    vector<point> points(n);
    set<point> s;

    for (int i = 0; i < n; i++) {
        cin >> points[i].x >> points[i].y;
        points[i].ind = i + 1;
        s.insert(points[i]);
    }

    sort(points.begin(), points.end());

    long long minSqr = (20002) * (20002);

    vector<int> ans;

    for (int left_down_i = 0; left_down_i < n - 1; left_down_i++) {
        for (int right_up_j = left_down_i + 1; right_up_j < n; right_up_j++) {
            long long left_up_x = points[left_down_i].x;
            long long left_up_y = points[right_up_j].y;

            long long right_down_x = points[right_up_j].x;
            long long right_down_y = points[left_down_i].y;

            point left_up = { left_up_x, left_up_y };
            point right_down = { right_down_x, right_down_y };
            auto it_l = s.find(left_up);
            auto it_r = s.find(right_down);
            if (it_l != s.end() && it_r != s.end()) {
                if (abs((left_up_y - right_down_y) * (right_down_x - left_up_x) < minSqr && (left_up_y - right_down_y) * (right_down_x - left_up_x)) != 0) {
                    minSqr = (left_up_y - right_down_y) * (right_down_x - left_up_x);
                    ans = { points[left_down_i].ind, points[right_up_j].ind, it_l->ind, it_r->ind };
                }
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        cout << ans[i] << " ";
    }
}