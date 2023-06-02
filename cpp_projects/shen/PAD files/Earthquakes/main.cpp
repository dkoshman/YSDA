#include <iostream>
#include <sstream>
#include <fstream>
#include <map>

using namespace std;

struct Time {
    int hour;
    int mins;
    int seconds;
    int toSeconds() const {
        return ((hour*60) + mins)*60 + seconds;
    }
    bool operator<(const Time& t2) const {
        return this->toSeconds() < t2.toSeconds();
    }

};

istream& operator>>(istream &in, Time& t){
    string s;
    stringstream ss;

    getline(in, s, ':');
    if (s[0] == '0')
        s.erase(0, 1);
    t.hour = stoi(s);
    getline(in, s, ':');
    if (s[0] == '0')
        s.erase(0, 1);
    t.mins = stoi(s);
    getline(in, s, ':');
    if (s[0] == '0')
        s.erase(0, 1);
    t.seconds = stoi(s);
    return in;
}

ostream& operator<<(ostream& out, const Time& t){
    out << t.hour << ':' << t.mins << ':' << t.seconds;
    return out;
}

struct Earthquake {
    double lat;
    double lon;
    double depth;
    double magnitude;
    string location;
};

ostream& operator<<(ostream& out, const Earthquake& q) {
    out << "lat=" << q.lat << ' ';
    out << "lon=" << q.lon << ' ';
    out << "depth=" << q.depth << ' ';
    out << "location=" << q.location << endl;
    return out;
}

void fillMap(multimap<Time, Earthquake>& m){
    const std::string INP_FILE_NAME = "../Earthquakes/earthquakes.csv";
    std::ifstream in;
    string buffer;
    string buffer2;
    stringstream ss;
    stringstream st;
    double d;
    Time t;
    Earthquake q;

    in.open(INP_FILE_NAME);
    getline(in, buffer);
    while (in.good()){
        getline(in, buffer);
        ss.clear();
        ss.str(buffer);
        getline(ss, buffer2, ',');
        st.clear();
        st.str(buffer2);
        st >> t;
        getline(ss, buffer, ',');
        st.clear();
        st.str(buffer);
        ss >> d;
        q.lat = d;
        getline(ss, buffer, ',');
        st.clear();
        st.str(buffer);
        ss >> d;
        q.lon = d;
        getline(ss, buffer, ',');
        st.clear();
        st.str(buffer);
        ss >> d;
        q.depth = d;
        getline(ss, buffer, ',');
        st.clear();
        st.str(buffer);
        ss >> d;
        q.magnitude = d;
        getline(ss, buffer, ',');
        q.location = buffer;
        m.insert(make_pair(t, q));
    }
}

void queryEarthquakes(multimap<Time, Earthquake>& m, const Time& tstart, const Time& tend, string x){
    bool match = false;
    for (pair<Time, Earthquake> p : m){
        if (tstart < p.first
            && p.first < tend
            && p.second.location.find(x) != string::npos){
            cout << p.second;
            match = true;
        }
    }
    if (not match)
        cout << "No matches." << endl;
}

int main()
{
    Time tstart, tend;
    string ts = "20:05:00";
    string te = "23:59:59";
    string x = "Hawaii";
    stringstream ss(ts);

    ss >> tstart;
    ss.clear();
    ss.str(te);
    ss >> tend;
    multimap<Time, Earthquake> m;
    fillMap(m);
    queryEarthquakes(m, tstart, tend, x);
    return 0;
}
