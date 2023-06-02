#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <set>


struct Applicant {
    std::string name, surname;
    unsigned score{}, day_birth{}, month_birth{}, year_birth{};
    std::vector<std::string> uni_list;
};

struct ApplicantPred {
    bool considered_score{};

    ApplicantPred(bool considered_score = false) : considered_score{considered_score} {}

    bool operator() (const Applicant& lhv, const Applicant& rhv) const {
        if (considered_score) {
            if (lhv.score > rhv.score)
                return true;
            if (lhv.score < rhv.score)
                return false;
            if (lhv.year_birth < rhv.year_birth)
                return true;
            if (lhv.year_birth > rhv.year_birth)
                return false;
            if (lhv.month_birth < rhv.month_birth)
                return true;
            if (lhv.month_birth > rhv.month_birth)
                return false;
            if (lhv.day_birth < rhv.day_birth)
                return true;
            if (lhv.day_birth > rhv.day_birth)
                return false;
            if (lhv.surname < rhv.surname)
                return true;
            if (lhv.surname > rhv.surname)
                return false;
            if (lhv.name < rhv.name)
                return true;
            if (lhv.name > rhv.name)
                return false;
            return true;
        }
        if (lhv.surname < rhv.surname)
            return true;
        if (lhv.surname > rhv.surname)
            return false;
        if (lhv.name < rhv.name)
            return true;
        if (lhv.name > rhv.name)
            return false;
        if (lhv.year_birth < rhv.year_birth)
            return true;
        if (lhv.year_birth > rhv.year_birth)
            return false;
        if (lhv.month_birth < rhv.month_birth)
            return true;
        if (lhv.month_birth > rhv.month_birth)
            return false;
        if (lhv.day_birth < rhv.day_birth)
            return true;
        if (lhv.day_birth > rhv.day_birth)
            return false;
        return true;
    }
};

using std::cin, std::cout;
typedef std::map<std::string, unsigned> UniCountMap;
typedef std::vector<Applicant> AppVector;
typedef std::map<std::string, std::set<Applicant, ApplicantPred>> UniAppMap;

UniCountMap read_universities(std::istream &in) {
    int n{};
    in >> n;
    UniCountMap uni_map;
    for (int i{}; i < n; ++i) {
        std::string s;
        in >> s;
        unsigned u{};
        in >> u;
        uni_map.insert(std::make_pair(s, u));
    }
    return uni_map;
}

AppVector read_applicants(std::istream &in) {
    int n{};
    in >> n;
    AppVector apps;
    for (int i = 0; i < n; ++i) {
        Applicant A;
        std::string s;
        in >> s;
        A.name = s;
        in >> s;
        A.surname = s;
        unsigned day{}, month{}, year{}, score{}, uni_count{};
        in >> day >> month >> year >> score >> uni_count;
        A.day_birth = day;
        A.month_birth = month;
        A.year_birth = year;
        A.score = score;
        for (unsigned j{}; j < uni_count; ++j) {
            in >> s;
            A.uni_list.push_back(s);
        }
        apps.push_back(A);
    }
    return apps;
}

UniAppMap distribute_students(UniCountMap &uni_count_map, AppVector &apps) {
    UniAppMap uni_app_map;
    for (auto &app : apps) {
        for (auto &uni : app.uni_list) {
            if (uni_count_map[uni] > 0) {
                --uni_count_map[uni];
                uni_app_map[uni].insert(app);
                break;
            }
        }
    }
    return uni_app_map;
}

void print_app_map(UniAppMap &uni_app_map, UniCountMap &uni_count_map) {
    for (auto &uni : uni_count_map) {
        cout << uni.first;
        for (auto &app : uni_app_map[uni.first])
            cout << '\t' << app.name << ' ' << app.surname;
        cout << std::endl;
    }
}

int main()
{
//    std::ifstream in("in.txt");
//    cin.rdbuf(in.rdbuf());
    UniCountMap uni_count_map = read_universities(cin);
    AppVector apps = read_applicants(cin);
    ApplicantPred app_pred(true);
    std::sort(std::begin(apps), std::end(apps), app_pred);
    UniAppMap uni_app_map = distribute_students(uni_count_map, apps);
    print_app_map(uni_app_map, uni_count_map);
    return 0;
}
