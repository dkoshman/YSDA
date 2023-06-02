#include <iostream>
#include <utility>

struct Specialization {
    std::string value;

    explicit Specialization(std::string value) : value(std::move(value)) {
    }
};
struct Course {
    std::string value;

    explicit Course(std::string value) : value(std::move(value)) {
    }
};
struct Week {
    std::string value;

    explicit Week(std::string value) : value(std::move(value)) {
    }
};

struct LectureTitle {
    std::string specialization;
    std::string course;
    std::string week;
    LectureTitle(Specialization s, Course c, Week w)
        : specialization(s.value), course(c.value), week(w.value) {
    }
};