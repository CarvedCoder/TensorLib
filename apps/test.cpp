#include <filesystem>
#include <tensorlib/ops.h>
#include <tensorlib/tensor.h>
#include <tensorlib/tokenizer.h>
#include <tensorlib/trainer.h>
constexpr int vocab_size = 2048;

int main(int argc, char* argv[]) {
    namespace tokenizer = tensorlib::tokenizer;
    std::filesystem::path Data = "../input.txt";
    tokenizer::Tokenizer tok;
    tokenizer::BPETrainer trainer;
    tok = trainer.train(Data, vocab_size);
}
