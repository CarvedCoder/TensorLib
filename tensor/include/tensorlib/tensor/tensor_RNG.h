#include <random>
class TensorRNG {
  public:
    static std::mt19937 &engine();
    static void setSeed(uint32_t seed);
};
