#include <cstdlib>

int llama_cli(int argc, char ** argv);

int main(int argc, char ** argv) {
#if defined(_WIN32)
    if (_putenv_s("GGML_VK_PQ2_BONSAI_TERNARY_DEDICATED", "1") != 0) {
        return 1;
    }
#else
    if (setenv("GGML_VK_PQ2_BONSAI_TERNARY_DEDICATED", "1", 1) != 0) {
        return 1;
    }
#endif

    return llama_cli(argc, argv);
}
