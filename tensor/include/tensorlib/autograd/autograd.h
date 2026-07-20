#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <tensorlib/tensor.h>
#include <unordered_map>
#include <vector>

class Tensor;

struct Node {
    std::vector<std::shared_ptr<TensorImpl>> inputs;
    std::weak_ptr<TensorImpl> output;
    std::string op_name;
    bool is_leaf = false;
    size_t id{};
    std::function<void(const float* upstream_grad, size_t size)> backward_fn;
    size_t output_size{};
};

class Autograd {
  private:
    inline static std::unordered_map<size_t, std::weak_ptr<Node>> registry;
    inline static size_t next_id{};

  public:
    static std::shared_ptr<Node> makeNode(std::vector<std::shared_ptr<TensorImpl>>& inputs,
                                          std::shared_ptr<TensorImpl>& output, std::string op_name,
                                          std::function<void(const float*, size_t)> backward_fn,
                                          bool is_leaf = false);
    void clearExpired();

    static const std::unordered_map<size_t, std::weak_ptr<Node>>& getRegistry();
};

#endif // !AUTOGRAD_H
