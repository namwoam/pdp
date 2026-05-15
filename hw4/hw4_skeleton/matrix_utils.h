
#include <random>
using namespace std;

mt19937& get_rng() {
    static random_device rd;
    static mt19937 gen(rd());
    return gen;
}

void randomize_matrix(float *mat, int N, float min_val = -5.0f, float max_val = 5.0f) {
    auto& gen = get_rng();
    uniform_real_distribution<float> dis(min_val, max_val);

    for (int i = 0; i < N; i++) {
        mat[i] = dis(gen);
    }
}

bool verify_matrix(float* mat_ref, float* mat_out, int N) {
    const double threshold = 1e-2;

    double diff = 0.0;
    for (int i = 0; i < N; i++) {
        diff = fabs(mat_ref[i] - mat_out[i]);
        if (isnan(diff) || diff > threshold) {
            printf("wrong at idx: %d with diff: %f\n", i, diff);
            return false;
        }
    }
    return true;
}