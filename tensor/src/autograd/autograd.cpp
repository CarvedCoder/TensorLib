#include <functional>
#include <memory>
#include <tensorlib/autograd.h>
#include <utility>

std::shared_ptr<Node> Autograd::makeNode(std::vector<std::shared_ptr<TensorImpl>>& inputs,
                                         std::shared_ptr<TensorImpl>& output, std::string op_name,
                                         std::function<void(const float*, size_t)> backward_fn,
                                         bool is_leaf) {
    auto node = std::make_shared<Node>();

    node->id = next_id++;
    node->inputs = inputs;
    node->output = output;
    node->op_name = std::move(op_name);
    node->backward_fn = std::move(backward_fn);
    node->is_leaf = is_leaf;
    node->output_size = output->m_total_size;

    registry[node->id] = node;

    return node;
}

void Autograd::clearExpired() {
    for (auto it = registry.begin(); it != registry.end();) {
        it = it->second.expired() ? registry.erase(it) : it++;
    }
}

const std::unordered_map<size_t, std::weak_ptr<Node>>& Autograd::getRegistry() {
    return registry;
}
