#include <algorithm>
#include <iostream>
#include <vector>
using std::cout, std::endl, std::cin, std::vector;

//struct Image {
//    double quality;
//    double freshness;
//    double rating;
//};
//
//struct Params {
//    double a;
//    double b;
//    double c;
//};

class FunctionPart {
public:
    FunctionPart(char c, double param) : operation_{c}, param_{param} {
    }
    double Apply(double value) const {
        if (operation_ == '+') {
            return value + param_;
        } else if (operation_ == '-') {
            return value - param_;
        } else if (operation_ == '*') {
            return value * param_;
        } else if (operation_ == '/') {
            return value / param_;
        }
    }
    void Invert() {
        if (operation_ == '+') {
            operation_ = '-';
        } else if (operation_ == '-') {
            operation_ = '+';
        } else if (operation_ == '*') {
            operation_ = '/';
        } else if (operation_ == '/') {
            operation_ = '*';
        }
    }

private:
    char operation_;
    double param_;
};

class Function {
public:
    void AddPart(char c, double param) {
        function_parts_.push_back({c, param});
    }
    void Invert() {
        for (auto& part : function_parts_) {
            part.Invert();
        }
        std::reverse(function_parts_.begin(), function_parts_.end());
    }
    double Apply(double value) const {
        for (auto& part : function_parts_) {
            value = part.Apply(value);
        }
        return value;
    }

private:
    vector<FunctionPart> function_parts_;
};

//Function MakeWeightFunction(const Params& params, const Image& image) {
//    Function function;
//    function.AddPart('*', params.a);
//    function.AddPart('-', image.freshness * params.b);
//    function.AddPart('+', image.rating * params.c);
//    return function;
//}
//
//double ComputeImageWeight(const Params& params, const Image& image) {
//    Function function = MakeWeightFunction(params, image);
//    return function.Apply(image.quality);
//}
//
//double ComputeQualityByWeight(const Params& params, const Image& image, double weight) {
//    Function function = MakeWeightFunction(params, image);
//    function.Invert();
//    return function.Apply(weight);
//}
//
//int main() {
//    Image image = {10, 2, 6};
//    Params params = {4, 2, 6};
//    cout << ComputeImageWeight(params, image) << endl;
//    cout << ComputeQualityByWeight(params, image, 52) << endl;
//    return 0;
//}
