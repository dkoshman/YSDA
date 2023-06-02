#include <iostream>
#include <memory>
#include <string>


class Expression {
public:
    virtual ~Expression() {}

    virtual int evaluate() const = 0;
    virtual std::string toString() const = 0;
};

using ExpressionPtr = std::shared_ptr<Expression>;

class ConstExp : public Expression {
public:
    int val;
    ConstExp(int val) : val(val) {}
    ~ConstExp() override = default;

    int evaluate() const override {return val;}
    std::string toString() const override {return std::to_string(val);}
};

ExpressionPtr Const(int val) {
    return std::make_shared<ConstExp>(val);
}

class SumExp : public Expression {
public:
    ExpressionPtr lhv, rhv;
    SumExp(ExpressionPtr lhv, ExpressionPtr rhv) : lhv(lhv), rhv(rhv) {}
    ~SumExp() override = default;

    int evaluate() const override {return lhv->evaluate() + rhv->evaluate();}
    std::string toString() const override {return lhv->toString() + " + " + rhv->toString();}
};

ExpressionPtr Sum(ExpressionPtr lhv, ExpressionPtr rhv) {
    return std::make_shared<SumExp>(lhv, rhv);
}

class ProductExp : public Expression {
public:
    ExpressionPtr lhv, rhv;
    ProductExp(ExpressionPtr lhv, ExpressionPtr rhv) : lhv(lhv), rhv(rhv) {}
    ~ProductExp() override = default;

    int evaluate() const override {return lhv->evaluate() * rhv->evaluate();}
    std::string toString() const override {
        std::string s;
        if (typeid(*lhv.get()) == typeid(SumExp))
            s += '(' + lhv->toString() + ')';
        else
            s += lhv->toString();
        s += " * ";
        if (typeid(*rhv.get()) == typeid(SumExp))
            s += '(' + rhv->toString() + ')';
        else
            s += rhv->toString();
        return s;
    }
};

ExpressionPtr Product(ExpressionPtr lhv, ExpressionPtr rhv) {
    return std::make_shared<ProductExp>(lhv, rhv);
}


int main()
{
    ExpressionPtr ex1 = Sum(Product(Const(3), Const(4)), Product(Const(5), Const(2)));
    std::cout << ex1->toString() << "\n";  // 3 * 4 + 5
    std::cout << ex1->evaluate() << "\n";  // 17

    ExpressionPtr ex2 = Product(Const(6), ex1);
    std::cout << ex2->toString() << "\n";  // 6 * (3 * 4 + 5)
    std::cout << ex2->evaluate() << "\n";  // 102

    ExpressionPtr ex = Product(Sum(Const(5), Sum(Const(1), Const(9))), Sum(Const(5), ex2));
    std::cout << ex.use_count() << std::endl;
    std::cout << ex->toString() << "\n";  // 3 * 4 + 5
    std::cout << typeid(ex.get()).name() << std::endl;
    std::cout << ex->evaluate() << "\n";  // 17
    std::cout << ex.use_count() << std::endl;
}
