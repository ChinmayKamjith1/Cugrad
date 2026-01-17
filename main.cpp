#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <algorithm> // For sorting/unique if you really want Set behavior
#include <cmath>

class Value : public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad;
    std::function<void()> _backward;
    std::vector<std::shared_ptr<Value>> _prev;
    std::string _op;
    std::string label;
    
    Value(double val, std::vector<std::shared_ptr<Value>> children = {}, std::string op = "", std::string lbl = ""){
        data = val;
        grad = 0;
        _backward = [](){};
        _prev = children;
        _op = op;
        label = lbl;
    }

    // The Engine: Topological Sort + Backprop
    void backward() {
        // 1. Prepare the lists
        std::vector<std::shared_ptr<Value>> topo;
        std::vector<std::shared_ptr<Value>> visited;

        // 2. Define the recursive helper (Lambda)
        std::function<void(std::shared_ptr<Value>)> build_topo = 
            [&](std::shared_ptr<Value> v) {
                
                // Check if already visited (Pointer comparison)
                for (auto& x : visited) {
                    if (x == v) return; 
                }
                visited.push_back(v);

                // Recursively visit all children (parents in the graph)
                for (auto& child : v->_prev) {
                    build_topo(child);
                }

                // Add self to the topological order (Post-order)
                topo.push_back(v);
            };

        // 3. Start the recursion from THIS node
        build_topo(shared_from_this());

        // 4. THE SPARK: Set the gradient of the final node to 1.0
        grad = 1.0;

        // 5. Run backward pass in REVERSE topological order
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }

};

//addition
std::shared_ptr<Value> operator+(std::shared_ptr<Value> self, std::shared_ptr<Value> other){
    auto out = std::make_shared<Value>(
        self->data + other->data,
        std::vector<std::shared_ptr<Value>>{self, other},
        "+"
    );

    out->_backward = [self, other, out](){
        self->grad += 1 * out->grad;
        other->grad += 1 * out->grad;
    };

    return out;
}

std::shared_ptr<Value> operator+(std::shared_ptr<Value> self, double other){
    auto other_val = std::make_shared<Value>(other);
    return self + other_val;
}

std::shared_ptr<Value> operator+(double self_val, std::shared_ptr<Value> other){
    auto self = std::make_shared<Value>(self_val);
    return self + other;
}

//multiply
std::shared_ptr<Value> operator*(std::shared_ptr<Value> self, std::shared_ptr<Value> other){
    auto out = std::make_shared<Value>(
        self->data * other->data,
        std::vector<std::shared_ptr<Value>> {self, other},
        "*"
    );

    out->_backward = [self, other, out](){
        self->grad += other->data * out->grad;
        other->grad += self->data * out->grad;
    };

    return out;
}

std::shared_ptr<Value> operator*(std::shared_ptr<Value> self, double other){
    auto other_val = std::make_shared<Value>(other);
    return self * other_val;
}

std::shared_ptr<Value> operator*(double self, std::shared_ptr<Value> other){
    auto self_val = std::make_shared<Value>(self);
    return self_val * other;
}

//powers
std::shared_ptr<Value> pow(std::shared_ptr<Value> self, double exponent){
    double data = std::pow(self->data, exponent);
    auto out = std::make_shared<Value>(
        data,
        std::vector<std::shared_ptr<Value>> {self},
        "**" + std::to_string(exponent)
    );

    out->_backward = [self, exponent, out](){
        double derivative = exponent * std::pow(self->data, (exponent - 1));
        self->grad += derivative * out->grad;   
    };

    return out;
}

//division
std::shared_ptr<Value> operator/(std::shared_ptr<Value> self, std::shared_ptr<Value> other){
    return self * pow(other, -1);
}

std::shared_ptr<Value> operator/(std::shared_ptr<Value> self, double other){
    return self * (1/other);
}

std::shared_ptr<Value> operator/(double self, std::shared_ptr<Value> other){
    auto self_val = std::make_shared<Value>(self);
    return self_val * pow(other, -1);
}

//subtraction
std::shared_ptr<Value> operator-(std::shared_ptr<Value> self){
    return self * -1;
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> self, std::shared_ptr<Value> other){
    return self + (-other);
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> self, double other){
    auto other_val = std::make_shared<Value>(other);
    return self + (-other_val); //prevent recursion by not doing self - other
}

std::shared_ptr<Value> operator-(double self, std::shared_ptr<Value> other){
    auto self_val = std::make_shared<Value>(self);
    return self_val + (-other);
}

//exp
std::shared_ptr<Value> exp(std::shared_ptr<Value> self){
    double data = std::exp(self->data);

    auto out = std::make_shared<Value> (
        data,
        std::vector<std::shared_ptr<Value>> {self},
        "exp"
    );

    out->_backward = [self, out](){
        self->grad += out->data * out->grad;
    };

    return out;
}

//tanh
std::shared_ptr<Value> tanh(std::shared_ptr<Value> self){
    double x = self->data;
    double data = std::tanh(x);

    auto out = std::make_shared<Value>(
        data,
        std::vector<std::shared_ptr<Value>> {self},
        "tanh"
    );

    out->_backward = [self, data, out](){
        self->grad += (1.0 - (data * data)) * out->grad; 
    };

    return out;
}

int main() {
    // 1. Define the Inputs (matching your screenshot)
    auto a = std::make_shared<Value>(2.0);
    auto b = std::make_shared<Value>(-3.0);
    auto c = std::make_shared<Value>(10.0);
    auto f = std::make_shared<Value>(-2.0);

    // 2. Forward Pass
    // e = a * b
    auto e = a * b;
    
    // d = e + c
    auto d = e + c;
    
    // L = d * f
    auto L = d * f;

    // 3. Backward Pass
    L->backward();

    // 4. Verify Results
    std::cout << "--- Forward Pass ---\n";
    std::cout << "L data: " << L->data << " (Expected: -8.0)\n\n";

    std::cout << "--- Backward Pass (Gradients) ---\n";
    std::cout << "L.grad: " << L->grad << " (Expected: 1.0)\n";
    std::cout << "f.grad: " << f->grad << " (Expected: 4.0)\n";
    std::cout << "d.grad: " << d->grad << " (Expected: -2.0)\n";
    std::cout << "e.grad: " << e->grad << " (Expected: -2.0)\n";
    std::cout << "c.grad: " << c->grad << " (Expected: -2.0)\n";
    std::cout << "b.grad: " << b->grad << " (Expected: -4.0)\n";
    std::cout << "a.grad: " << a->grad << " (Expected: 6.0)\n";

    return 0;
}