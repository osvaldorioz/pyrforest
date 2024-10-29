#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

//c++ -O3 -Wall -shared -std=c++20 -fPIC $(python3.12 -m pybind11 --includes) random_forest_optimized.cpp -o random_forest_module$(python3.12-config --extension-suffix)


namespace py = pybind11;
typedef std::vector<std::vector<double>> Dataset;

class DecisionTree {
public:
    DecisionTree(int max_depth = 5) : max_depth(max_depth) {}

    void fit(const Dataset& X, const std::vector<int>& y, int seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<> dist(0, X.size() - 1);
        
        // Entrenamiento simplificado
        majority_class = std::accumulate(y.begin(), y.end(), 0) > (y.size() / 2) ? 1 : 0;
    }

    int predict(const std::vector<double>& x) const {
        return majority_class;
    }

private:
    int max_depth;
    int majority_class;
};

class RandomForest {
public:
    RandomForest(int num_trees, int max_depth, int max_features)
        : num_trees(num_trees), max_depth(max_depth), max_features(max_features) {}

    void fit(const Dataset& X, const std::vector<int>& y) {
        trees.clear();
        trees.resize(num_trees);
        std::vector<std::thread> threads;

        for (int i = 0; i < num_trees; ++i) {
            threads.emplace_back([this, &X, &y, i]() {
                DecisionTree tree(max_depth);
                tree.fit(X, y, i + 42);
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    trees[i] = std::move(tree);
                }
            });
        }

        for (auto& th : threads) th.join();
    }

    int predict(const std::vector<double>& x) const {
        std::vector<int> predictions;
        for (const auto& tree : trees) {
            predictions.push_back(tree.predict(x));
        }

        int sum = std::accumulate(predictions.begin(), predictions.end(), 0);
        return (sum > trees.size() / 2) ? 1 : 0;
    }

private:
    int num_trees;
    int max_depth;
    int max_features;
    std::vector<DecisionTree> trees;
    mutable std::mutex mtx;
};

PYBIND11_MODULE(random_forest_module, m) {
    py::class_<RandomForest>(m, "RandomForest")
        .def(py::init<int, int, int>(), py::arg("num_trees"), py::arg("max_depth"), py::arg("max_features"))
        .def("fit", &RandomForest::fit, "Entrenamiento", py::arg("X"), py::arg("y"))
        .def("predict", &RandomForest::predict, "Prediccion usando Random Forest", py::arg("x"));
}
