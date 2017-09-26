#pragma once

#include "common/definitions.h"
#include "data/batch_generator.h"

namespace marian {
//This class takes care of per thread values of tau_ and batch-flexible-lr
//so that we can do warmup
class Scaler {
  private:
    Ptr<Config> options_;

    float min_batch_words;
    float max_batch_words;
    size_t num_batch_regain;
    float batch_lr_step;

    size_t max_tau;
    size_t tau_regain_batches;
    double tau_step;
    double next_increment;

    size_t current_batch;

    size_t current_tau;
    float current_batch_words;

    void init() {
      //How many batches need to pass for us to increment tau
      tau_step = std::max((double)tau_regain_batches / (double)(max_tau - 1), 1.0);
      next_increment = tau_step;
      if (tau_regain_batches == 1) { //Only do that if necessary
        current_tau = max_tau;
      } else {
        current_tau = 1;
      }

      //How many batches need to pass for us to decrement LR
      if (max_batch_words != min_batch_words && num_batch_regain != 1) { 
        batch_lr_step = num_batch_regain / (max_batch_words - min_batch_words);
        batch_lr_step = 1/batch_lr_step;
      } else {
        batch_lr_step = 0;
        max_batch_words = min_batch_words;
      }
      current_batch_words = max_batch_words;
    }


  public:
    Scaler(float min_batch_words_, float max_batch_words_, size_t num_batch_regain_, //For testing
      size_t max_tau_, size_t tau_regain_batches_) :
      min_batch_words(min_batch_words_),
      max_batch_words(max_batch_words_),
      num_batch_regain(num_batch_regain_),
      max_tau(max_tau_),
      tau_regain_batches(tau_regain_batches_),
      current_batch(0)
    {
      init();
    }
    Scaler(Ptr<Config> options) : 
      options_(options),
      min_batch_words(options->get<float>("batch-normal-words")),
      max_batch_words(options->get<float>("batch-max-words")),
      num_batch_regain(options->get<size_t>("batch-words-regain")),
      max_tau(options->get<size_t>("tau")),
      tau_regain_batches(options->get<size_t>("tau-max-regain")),
      current_batch(0) {
        init();
      }

    void newBatch() {
      current_batch++;
      if (current_batch >= next_increment && current_tau < max_tau) {
        current_tau++;
        next_increment += tau_step;
      }

      if (current_batch_words > min_batch_words) {
        current_batch_words = std::max(current_batch_words - batch_lr_step, min_batch_words);
      }
    }

    size_t getNewTau() {
      return current_tau;
    }

    float getNewBatchLR() {
      return current_batch_words;
    }

};
} //namespace
