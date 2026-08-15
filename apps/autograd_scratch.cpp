#include "tensorlib/autograd/autograd.h"
#include "tensorlib/tensor/tensor.h"
#include <algorithm>
#include <initializer_list>
#include <memory>
#include <tensorlib/ops.h>
#include <tensorlib/tensor.h>
#include <unordered_set>
#include <vector>
/*
int id{};

struct AutogradNode {
    std::vector<float> parents;
    std::string mode{};
    float output;
    int _id;
    explicit AutogradNode(std::vector<float>& inputs, const float& output, const std::string& mode)
        : parents(inputs), output(output), mode(mode) {
        _id = id++;
    }
};

std::shared_ptr<AutogradNode> makeAutogradNode(const std::initializer_list<float>& inputs,
                                               const float& output, const std::string& mode) {
    std::vector<float> parents{inputs};
    auto node = std::make_shared<AutogradNode>(parents, output, mode);
    return node;
}

std::vector<std::shared_ptr<Node>> topoSort(const std::shared_ptr<TensorImpl>& root) {
    std::vector<std::shared_ptr<Node>> order;
    std::unordered_set<size_t> visited;
    std::function<void(const std::shared_ptr<floatImpl>&)> dfs =
        [&](const std::shared_ptr<floatImpl>& tensor) {
            const auto& node = tensor->m_grad_fn;
            if (!node || !visited.insert(node->id).second)
                return;
            for (const auto& input : node->inputs)
                dfs(input);

            order.push_back(node);
        };
    dfs(root);
    return order;
}

void enableGrad(const float& t) {
    t.getImpl()->m_require_grad = true;
    t.getImpl()->ensureGrad();
}

using namespace floatOps;
*/

int main(int argc, char* argv[]) {
    /*
    float a = float ::createScalar(2.0f);
    float b = float ::createScalar(3.0f);
    enableGrad(a);
    enableGrad(b);
    float c = a + b;
    float d = c + a;
    auto node1 = makeAutogradNode({a, b}, c, "add");
    auto node2 = makeAutogradNode({c, a}, d, "add");
    auto order = Autograd::topoSort(d.getImpl());
    std::reverse(order.begin(), order.end());
    for (const auto node : order) {
    }
    */
    return 0;
}
