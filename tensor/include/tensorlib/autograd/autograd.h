#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <cstddef>
#include <memory>
#include <string>
#include <tensorlib/tensor.h>
#include <unordered_map>
#include <vector>

class Tensor;

struct Node {
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::string op_name;
    bool is_leaf = false;
    size_t id;
};

class Autograd {
  private:
    inline static std::unordered_map<size_t, std::weak_ptr<Node>> registry;
    inline static size_t next_id{};

  public:
    static std::shared_ptr<Node> makeNode(std::vector<std::shared_ptr<Tensor>>& Nodes,
                                          std::string& op_name, bool is_leaf = false);
    static void clearExpired();

    const std::unordered_map<size_t, std::weak_ptr<Node>> getRegistry() const;
};

#endif // !AUTOGRAD_H
