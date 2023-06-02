#include <iostream>
#include <string>
#include <map>
#include <set>
#include <iomanip>
#include <sstream>

using namespace std;
// Реализуйте функции и методы классов и при необходимости добавьте свои

class Date {
public:
    int GetYear() const {
        return year;
    }
    int GetMonth() const {
        return month;
    }
    int GetDay() const {
        return day;
    }
    Date(int year, int month, int day) : year{year}, month{month}, day{day} {
    }

private:
    int year;
    int month;
    int day;
};

bool operator<(const Date& lhs, const Date& rhs) {
    if (lhs.GetYear() == rhs.GetYear()) {
        if (lhs.GetMonth() == rhs.GetMonth()) {
            return lhs.GetDay() < rhs.GetDay();
        } else {
            return lhs.GetMonth() < rhs.GetMonth();
        }
    } else {
        return lhs.GetYear() < rhs.GetYear();
    }
}

class Database {
public:
    void AddEvent(const Date& date, const string& event) {
        db_[date].insert(event);
    }
    bool DeleteEvent(const Date& date, const string& event) {
        bool success = db_[date].count(event);
        db_[date].erase(event);
        return success;
    }
    int DeleteDate(const Date& date) {
        int deletions = db_[date].size();
        db_[date].clear();
        return deletions;
    }

    void Find(const Date& date) const {
        if (db_.count(date)) {
            for (const auto& event : db_.at(date)) {
                cout << event << endl;
            }
        }
    }

    void Print() const {
        for (const auto& [date, events] : db_) {
            for (const auto& event : events) {
                cout << fixed << setw(4) << setfill('0') << date.GetYear() << '-';
                cout << fixed << setw(2) << setfill('0') << date.GetMonth() << '-';
                cout << fixed << setw(2) << setfill('0') << date.GetDay() << ' ';
                cout << event << endl;
            }
        }
    }

private:
    map<Date, set<string>> db_;
};

Date GetDate(const string& input) {
    stringstream stream(input);
    char delimiter_first, delimiter_second, last_char;
    int year, month, day;
    bool ok = bool(stream >> year >> delimiter_first >> month >> delimiter_second >> day);
    if (not ok || delimiter_first != '-' || delimiter_second != '-' || stream >> last_char) {
        throw invalid_argument("Wrong date format: " + input);
    }
    if (month < 1 || month > 12) {
        throw invalid_argument("Month value is invalid: " + to_string(month));
    }
    if (day < 1 || day > 31) {
        throw invalid_argument("Day value is invalid: " + to_string(day));
    }
    return Date{year, month, day};
}

int main() {
    Database db;

    string line;
    while (getline(cin, line)) {
        if (line.empty()) {
            continue;
        }
        // Считайте команды с потока ввода и обработайте каждую
        stringstream stream(line);
        string command, second_word, third_word;
        stream >> command >> second_word >> third_word;
        try {
            if (command == "Add") {
                Date date = GetDate(second_word);
                db.AddEvent(date, third_word);
            } else if (command == "Del") {
                Date date = GetDate(second_word);
                if (not third_word.empty()) {
                    if (db.DeleteEvent(date, third_word)) {
                        cout << "Deleted successfully" << endl;
                    } else {
                        cout << "Event not found" << endl;
                    }
                } else {
                    int deletions = db.DeleteDate(date);
                    cout << "Deleted " << deletions << " events" << endl;
                }
            } else if (command == "Find") {
                Date date = GetDate(second_word);
                db.Find(date);
            } else if (command == "Print") {
                db.Print();
            } else {
                cout << "Unknown command: " << command << endl;
            }
        } catch (invalid_argument& e) {
            cout << e.what() << endl;
        }
    }

    return 0;
}