//
// lite.ai.toolkit unified benchmark / timing utility (header-only, backend-agnostic)
//
// Goals:
//   * cross-platform, header-only, no dependency on any specific inference backend;
//   * CPU stages timed with std::chrono; GPU stages timed with cudaEvent
//     (so asynchronous calls are measured correctly, not via wall-clock);
//   * aggregate many samples into mean/p50/p90/p99/min/max and end-to-end FPS,
//     with optional CSV export.
//
// Typical usage:
//   lite::bench::Profiler prof;
//   for (int i = 0; i < N; ++i) {
//     lite::bench::CpuTimer total; total.start();
//     { LITE_CPU_SCOPE(prof, "preprocess");  /* ... */ }
//     { LITE_GPU_SCOPE(prof, "inference", stream); /* enqueueV3 ... */ }
//     { LITE_CPU_SCOPE(prof, "postprocess"); /* ... */ }
//     prof.tick(total.stop_ms());     // record one full iteration for FPS
//   }
//   prof.report("FaceFusion pipeline");
//   prof.to_csv("bench_facefusion.csv");
//
#ifndef LITE_AI_TOOLKIT_BENCH_PROFILER_H
#define LITE_AI_TOOLKIT_BENCH_PROFILER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

#if defined(ENABLE_TENSORRT) || defined(__CUDACC__)
#include <cuda_runtime.h>
#define LITE_BENCH_WITH_CUDA 1
#endif

namespace lite {
namespace bench {

// Aggregated stats for a single stage (unit: ms)
struct Stat {
  std::size_t calls = 0;
  double mean = 0, p50 = 0, p90 = 0, p99 = 0, min = 0, max = 0;

  static Stat from(std::vector<double> v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.calls = v.size();
    s.min = v.front();
    s.max = v.back();
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    s.p50 = percentile(v, 50);
    s.p90 = percentile(v, 90);
    s.p99 = percentile(v, 99);
    return s;
  }

 private:
  // v must be sorted ascending; nearest-rank percentile
  static double percentile(const std::vector<double> &v, double p) {
    if (v.empty()) return 0.0;
    double rank = (p / 100.0) * static_cast<double>(v.size() - 1);
    auto idx = static_cast<std::size_t>(std::llround(rank));
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
  }
};

class Profiler {
 public:
  // Record one sample (ms) for a stage
  void add(const std::string &stage, double ms) {
    auto it = samples_.find(stage);
    if (it == samples_.end()) {
      order_.push_back(stage);
      samples_[stage].push_back(ms);
    } else {
      it->second.push_back(ms);
    }
  }

  // Record one full-iteration end-to-end latency (used to compute FPS)
  void tick(double ms) { add(kTotal, ms); }

  void clear() {
    order_.clear();
    samples_.clear();
  }

  // Print an aligned table; the caller is responsible for excluding warmup samples
  void report(const std::string &title, std::ostream &os = std::cout) const {
    os << "\n==================== Benchmark: " << title
       << " ====================\n";
    os << std::left << std::setw(18) << "stage" << std::right << std::setw(8)
       << "calls" << std::setw(11) << "mean(ms)" << std::setw(11) << "p50"
       << std::setw(11) << "p90" << std::setw(11) << "p99" << std::setw(11)
       << "min" << std::setw(11) << "max" << "\n";
    os << std::string(92, '-') << "\n";
    for (const auto &stage : order_) {
      if (stage == kTotal) continue;
      print_row(os, stage, Stat::from(samples_.at(stage)));
    }
    auto it = samples_.find(kTotal);
    if (it != samples_.end()) {
      os << std::string(92, '-') << "\n";
      Stat t = Stat::from(it->second);
      print_row(os, "TOTAL", t);
      if (t.mean > 0.0)
        os << "  -> throughput: " << std::fixed << std::setprecision(2)
           << (1000.0 / t.mean) << " FPS (by mean), " << (1000.0 / t.p50)
           << " FPS (by p50)\n";
    }
    os << std::string(92, '=') << "\n";
  }

  // Export CSV (stage,calls,mean,p50,p90,p99,min,max)
  void to_csv(const std::string &path) const {
    std::ofstream f(path);
    if (!f) {
      std::cerr << "[profiler] cannot write CSV: " << path << std::endl;
      return;
    }
    f << "stage,calls,mean_ms,p50_ms,p90_ms,p99_ms,min_ms,max_ms\n";
    for (const auto &stage : order_) {
      Stat s = Stat::from(samples_.at(stage));
      const char *name = (stage == kTotal) ? "TOTAL" : stage.c_str();
      f << name << "," << s.calls << "," << s.mean << "," << s.p50 << ","
        << s.p90 << "," << s.p99 << "," << s.min << "," << s.max << "\n";
    }
    std::cout << "[profiler] CSV written: " << path << std::endl;
  }

  Stat stat(const std::string &stage) const {
    auto it = samples_.find(stage);
    return it == samples_.end() ? Stat{} : Stat::from(it->second);
  }

 private:
  static constexpr const char *kTotal = "__total__";

  static void print_row(std::ostream &os, const std::string &name,
                        const Stat &s) {
    os << std::left << std::setw(18) << name << std::right << std::setw(8)
       << s.calls << std::fixed << std::setprecision(3) << std::setw(11)
       << s.mean << std::setw(11) << s.p50 << std::setw(11) << s.p90
       << std::setw(11) << s.p99 << std::setw(11) << s.min << std::setw(11)
       << s.max << "\n";
  }

  std::vector<std::string> order_;
  std::unordered_map<std::string, std::vector<double>> samples_;
};

// ---------------- CPU timing (chrono) ----------------
class CpuTimer {
 public:
  void start() { t0_ = clock::now(); }
  double stop_ms() const {
    return std::chrono::duration<double, std::milli>(clock::now() - t0_).count();
  }

 private:
  using clock = std::chrono::high_resolution_clock;
  clock::time_point t0_ = clock::now();
};

// RAII: record CPU elapsed time into the profiler on scope exit
class ScopedCpuTimer {
 public:
  ScopedCpuTimer(Profiler &p, std::string stage)
      : prof_(p), stage_(std::move(stage)) {
    timer_.start();
  }
  ~ScopedCpuTimer() { prof_.add(stage_, timer_.stop_ms()); }

 private:
  Profiler &prof_;
  std::string stage_;
  CpuTimer timer_;
};

// Optional variant: when prof is nullptr it does nothing (zero overhead).
// Used to instrument library code that is off by default and only on when benchmarking.
class ScopedCpuTimerOpt {
 public:
  ScopedCpuTimerOpt(Profiler *p, std::string stage)
      : prof_(p), stage_(std::move(stage)) {
    if (prof_) timer_.start();
  }
  ~ScopedCpuTimerOpt() {
    if (prof_) prof_->add(stage_, timer_.stop_ms());
  }

 private:
  Profiler *prof_;
  std::string stage_;
  CpuTimer timer_;
};

#ifdef LITE_BENCH_WITH_CUDA
// ---------------- GPU timing (cudaEvent) ----------------
// Note: stop_ms() calls cudaEventSynchronize, so it serializes the stage; acceptable for benchmarking.
class CudaTimer {
 public:
  CudaTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start(cudaStream_t stream = nullptr) {
    stream_ = stream;
    cudaEventRecord(start_, stream_);
  }
  float stop_ms() {
    cudaEventRecord(stop_, stream_);
    cudaEventSynchronize(stop_);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }

 private:
  cudaEvent_t start_{}, stop_{};
  cudaStream_t stream_ = nullptr;
};

// RAII: record GPU elapsed time (on the given stream) into the profiler on scope exit
class ScopedCudaTimer {
 public:
  ScopedCudaTimer(Profiler &p, std::string stage, cudaStream_t stream = nullptr)
      : prof_(p), stage_(std::move(stage)) {
    timer_.start(stream);
  }
  ~ScopedCudaTimer() { prof_.add(stage_, timer_.stop_ms()); }

 private:
  Profiler &prof_;
  std::string stage_;
  CudaTimer timer_;
};
#endif  // LITE_BENCH_WITH_CUDA

}  // namespace bench
}  // namespace lite

// ---------------- convenience macros ----------------
#define LITE_BENCH_CONCAT_(a, b) a##b
#define LITE_BENCH_CONCAT(a, b) LITE_BENCH_CONCAT_(a, b)

// Time the current scope on the CPU and record it into the profiler
#define LITE_CPU_SCOPE(prof, name) \
  lite::bench::ScopedCpuTimer LITE_BENCH_CONCAT(_lite_cpu_scope_, __LINE__)((prof), (name))

// Optional variant taking a Profiler*; zero overhead when nullptr (for library instrumentation, off by default)
#define LITE_CPU_SCOPE_OPT(profptr, name) \
  lite::bench::ScopedCpuTimerOpt LITE_BENCH_CONCAT(_lite_cpu_scope_opt_, __LINE__)((profptr), (name))

#ifdef LITE_BENCH_WITH_CUDA
// Time the current scope on the GPU (given stream) and record it into the profiler
#define LITE_GPU_SCOPE(prof, name, stream) \
  lite::bench::ScopedCudaTimer LITE_BENCH_CONCAT(_lite_gpu_scope_, __LINE__)((prof), (name), (stream))
#endif

#endif  // LITE_AI_TOOLKIT_BENCH_PROFILER_H
