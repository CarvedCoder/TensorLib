#include "tensorlib/autograd/autograd.h"
#include <memory>
#include <tensorlib/autograd.h>

std::shared_ptr<Node> Autograd::makeNode(std::vector<std::shared_ptr<Tensor>>& Nodes,
                                         std::string& op_name, bool is_leaf) {
    auto node = std::make_shared<Node>();

    node->id = next_id++;
    node->inputs = Nodes;
    node->op_name = op_name;
    node->is_leaf = is_leaf;

    registry[node->id] = node;

    return node;
}

void Autograd::clearExpired() {
    for (auto it = registry.begin(); it != registry.end();) {
        it = it->second.expired() ? registry.erase(it) : it++;
    }
}

const std::unordered_map<size_t, std::weak_ptr<Node>> Autograd::getRegistry() const {
    return registry;
}
