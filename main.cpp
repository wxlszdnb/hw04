#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>

float frand() {
    return (float)rand() / RAND_MAX * 2 - 1;
}

constexpr int len_arr=4;

struct Star {
    float px[48], py[48], pz[48];
    float vx[48], vy[48], vz[48];
    float mass[48];
};

Star stars;

void init() {
    for (size_t j = 0; j < 48; ++j) {
            stars.px[j]=frand();
            stars.py[j]=frand();
            stars.pz[j]=frand();
            stars.vx[j]=frand();
            stars.vy[j]=frand();
            stars.vz[j]=frand();
            stars.mass[j]=frand()+1;

//        stars.push_back({
//            frand(), frand(), frand(),
//            frand(), frand(), frand(),
//            frand() + 1,
//        });
    }
}

constexpr float G = 0.001;
constexpr float eps = 0.001;
constexpr float dt = 0.01;

void step() {
#pragma omp simd
    for (size_t i = 0; i < 48; ++i) {
        for (size_t j = 0; j < 48; ++j) {
            float dx = stars.px[i] - stars.px[j];
            float dy = stars.py[i] - stars.py[j];
            float dz = stars.pz[i] - stars.pz[j];
            float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
            d2 *= std::sqrt(d2);
            auto G_dt_inv_d2= G * dt * (1 / d2);
            stars.vx[i] += dx * stars.mass[j] * G_dt_inv_d2;
            stars.vy[i] += dy * stars.mass[j] * G_dt_inv_d2;
            stars.vz[i] += dz * stars.mass[j] * G_dt_inv_d2;
        }
    }

#pragma omp simd
    for (size_t i = 0; i < 48; i++)  {
        stars.px[i] += stars.vx[i] * dt;
        stars.py[i] += stars.vy[i] * dt;
        stars.pz[i] += stars.vz[i] * dt;
    }
}

float calc() {
    float energy = 0;
    for (size_t i = 0; i < 48; i++) {
        float v2 = stars.vx[i] * stars.vx[i] + stars.vy[i] * stars.vy[i] + stars.vz[i] * stars.vz[i];
        energy += stars.mass[i] * v2 * 0.5F;
#pragma omp simd
        for (size_t j = 0; j < 48; j++) {
            float dx = stars.px[j] - stars.px[i];
            float dy = stars.py[j] - stars.py[i];
            float dz = stars.pz[j] - stars.pz[i];
            float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
            energy -= stars.mass[j] * stars.mass[i] * G / std::sqrt(d2) * 0.5F;
        }
    }
    return energy;
}

template <class Func>
long benchmark(Func const &func) {
    auto t0 = std::chrono::steady_clock::now();
    func();
    auto t1 = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    return dt.count();
}

int main() {
    init();
    printf("Initial energy: %f\n", calc());
    auto dt = benchmark([&] {
        for (int i = 0; i < 100000; i++)
            step();
    });
    printf("Final energy: %f\n", calc());
    printf("Time elapsed: %ld ms\n", dt);
    return 0;
}
