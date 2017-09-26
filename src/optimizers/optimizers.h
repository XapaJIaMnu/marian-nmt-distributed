#pragma once

#include <map>
#include <memory>

#include "common/config.h"
#include "graph/expression_graph.h"
#include "optimizers/clippers.h"
#include "tensors/tensor.h"
#include "training/training_state.h"

namespace marian {

class OptimizerBase : public TrainingObserver {
public:
  template <typename... Args>
  OptimizerBase(float eta, Args... args)
      : clipper_(Get(keywords::clip, nullptr, args...)), eta_(eta) {}

  void update(Ptr<ExpressionGraph> graph, float multiply_factor_ = 1.0f) {
    Tensor p = graph->params()->vals();
    Tensor g = graph->params()->grads();

    update(p, g, multiply_factor_);
  }

  void update(Tensor params, Tensor grads, float multiply_factor_ = 1.0f) {
    if(clipper_)
      clipper_->clip(grads);

    multiply_factor = multiply_factor_; //In case we want to add a multiply factor to our learning rate
    updateImpl(params, grads);
  }

  void actAfterEpoch(TrainingState& state) {
    eta_ = state.eta;
  }
  void actAfterBatches(TrainingState& state) {
    eta_ = state.eta;
  }
  void actAfterStalled(TrainingState& state) {
    eta_ = state.eta;
  }

  virtual Tensor getMT_() {
    return nullptr;
  }

  virtual Tensor getVT_() {
    return nullptr;
  }

  virtual void updateState(Ptr<OptimizerBase> localOpt, size_t shardSize_, int my_id) {
    return;
  }

  virtual void setB1(float newValue) {
    return;
  }

  virtual void setB2(float newValue) {
    return;
  }

  virtual void setEPS(float newValue) {
    return;
  }

protected:
  virtual void updateImpl(Tensor params, Tensor grads) = 0;

  Ptr<ClipperBase> clipper_;
  float eta_;
  float multiply_factor; //Compensates for larger batch
};

class Sgd : public OptimizerBase {
public:
  template <typename... Args>
  Sgd(float eta, Args... args) : OptimizerBase(eta, args...) {}

private:
  void updateImpl(Tensor params, Tensor grads);
};

// @TODO: Add serialization for historic gradients and parameters
class Adagrad : public OptimizerBase {
public:
  template <typename... Args>
  Adagrad(float eta, Args... args)
      : OptimizerBase(eta, args...), eps_(Get(keywords::eps, 1e-8, args...)) {}

private:
  void updateImpl(Tensor params, Tensor grads);

  float eps_;
  Ptr<TensorAllocator> alloc_;
  Tensor gt_;
};

// @TODO: Add serialization for historic gradients and parameters
// https://arxiv.org/pdf/1412.6980v8.pdf
class Adam : public OptimizerBase {
public:
  template <typename... Args>
  Adam(float eta, Args... args)
      : OptimizerBase(eta, args...),
        beta1_(Get(keywords::beta1, 0.91, args...)),
        beta2_(Get(keywords::beta2, 0.998, args...)),
        eps_(Get(keywords::eps, 1e-8, args...)),
        t_(0) {}

  void updateImpl(Tensor params, Tensor grads);
  void updateState(Ptr<OptimizerBase> localOpt, size_t shardSize_, int my_id);
  Tensor getMT_() {
    return mt_;
  }

  Tensor getVT_() {
    return vt_;
  }
  
  void setB1(float newValue) {
    beta1_ = newValue;
  }

  void setB2(float newValue) {
    beta2_ = newValue;
  }

  void setEPS(float newValue) {
    eps_ = newValue;
  }

private:
  float beta1_;
  float beta2_;
  float eps_;
  size_t t_;

  Ptr<TensorAllocator> mtAlloc_;
  Tensor mt_;
  Ptr<TensorAllocator> vtAlloc_;
  Tensor vt_;
};

template <class Algorithm, typename... Args>
Ptr<OptimizerBase> Optimizer(float eta, Args&&... args) {
  return Ptr<OptimizerBase>(new Algorithm(eta, args...));
}

Ptr<OptimizerBase> Optimizer(Ptr<Config> options);
Ptr<OptimizerBase> Optimizer(std::string opt, double lrate, double clipNorm = 0);
}
