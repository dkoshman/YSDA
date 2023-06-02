#include <iostream>
#include <string>
#include <vector>

struct Student {
    std::string name;
    std::string surname;
    int day;
    int month;
    int year;
};

int main() {
    int n_students;
    std::cin >> n_students;
    std::vector<Student> students(n_students);
    for (auto& student : students) {
        std::cin >> student.name;
        std::cin >> student.surname;
        std::cin >> student.day;
        std::cin >> student.month;
        std::cin >> student.year;
    }
    int n_requests;
    std::cin >> n_requests;
    for (int request = 0; request < n_requests; ++request) {
        std::string request_name;
        std::cin >> request_name;
        int request_parameter;
        std::cin >> request_parameter;
        --request_parameter;
        if (request_name == "name") {
            if (request_parameter < students.size() && request_parameter >= 0) {
                std::cout << students[request_parameter].name << ' '
                          << students[request_parameter].surname << '\n';
                continue;
            }
        }
        if (request_name == "date") {
            if (request_parameter < students.size() && request_parameter >= 0) {
                std::cout << students[request_parameter].day << '.'
                          << students[request_parameter].month << '.'
                          << students[request_parameter].year << '\n';
                continue;
            }
        }
        std::cout << "bad request\n";
    }
    return 0;
}
