//
// Created by TalkUHulk on 2024/4/25.
//

// reference by https://github.com/TalkUHulk/ddim_scheduler_cpp.git

#ifndef DDIM_SCHEDULER_CPP_DDIMSCHEDULER_HPP
#define DDIM_SCHEDULER_CPP_DDIMSCHEDULER_HPP

#include <iostream>
#include <vector>
#include <string>
namespace Scheduler {

#if defined(_MSC_VER)
#if defined(BUILDING_AIENGINE_DLL)
#define DDIM_PUBLIC __declspec(dllexport)
#elif defined(USING_AIENGINE_DLL)
#define DDIM_PUBLIC __declspec(dllimport)
#else
#define DDIM_PUBLIC
#endif
#else
#define DDIM_PUBLIC __attribute__((visibility("default")))
#endif
    
    struct DDIMMeta;
    class DDIM_PUBLIC DDIMScheduler {

    private:
        DDIMMeta* meta_ptr = nullptr;
        int num_inference_steps = 0;

    public:
        explicit DDIMScheduler(const std::string &config);

        ~DDIMScheduler();

        // Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        int set_timesteps(int num_inference_steps);

        void get_timesteps(std::vector<int> &dst);

        float get_init_noise_sigma() const;

        int step(std::vector<float> &model_output, const std::vector<int> &model_output_size,
                 std::vector<float> &sample, const std::vector<int> &sample_size,
                 std::vector<float> &prev_sample,
                 int timestep, float eta = 0.0, bool use_clipped_model_output = false);

        int add_noise(std::vector<float> &sample, const std::vector<int> &sample_size,
                      std::vector<float> &noise, const std::vector<int> &noise_size, int timesteps,
                      std::vector<float> &noisy_samples);
    private:
        float get_variance(int timestep, int prev_timestep);
    };
}


#endif //DDIM_SCHEDULER_CPP_DDIMSCHEDULER_HPP
