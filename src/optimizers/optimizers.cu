#include "optimizers.h"

#include "kernels/tensor_operators.h"
#include "kernels/thrust_functions.h"

namespace marian {
void Sgd::updateImpl(Tensor params, Tensor grads) {
  Element(_1 -= (multiply_factor*eta_) * _2, params, grads);

  cudaStreamSynchronize(0);
}

void Adagrad::updateImpl(Tensor params, Tensor grads) {
  if(!alloc_)
    alloc_ = New<TensorAllocator>(params->getDevice());

  if(!gt_) {
    int elements = params->size();
    alloc_->reserveExact(params->memory()->size());
    alloc_->allocate(gt_, {1, elements});
    gt_->set(0);
  }

  Element(_1 += (_2 * _2), gt_, grads);

  Element(_1 -= ((multiply_factor*eta_) / (Sqrt(_2) + eps_)) * _3, params, gt_, grads);

  cudaStreamSynchronize(0);
}

void Adam::updateImpl(Tensor params, Tensor grads) {
  if(!mtAlloc_)
    mtAlloc_ = New<TensorAllocator>(params->getDevice());
  if(!vtAlloc_)
    vtAlloc_ = New<TensorAllocator>(params->getDevice());

  if(!mt_) {
    int elements = params->size();
    mtAlloc_->reserveExact(params->memory()->size());
    mtAlloc_->allocate(mt_, {1, elements});
    mt_->set(0);

    vtAlloc_->reserveExact(params->memory()->size());
    vtAlloc_->allocate(vt_, {1, elements});
    vt_->set(0);
  }

  t_++;
  float denom1 = 1 - std::pow(beta1_, t_);
  float denom2 = 1 - std::pow(beta2_, t_);

  Element(_1 = (beta1_ * _1) + ((1 - beta1_) * _2), mt_, grads);
  Element(_1 = (beta2_ * _1) + ((1 - beta2_) * (_2 * _2)), vt_, grads);

  Element(_1 -= (multiply_factor*eta_) * (_2 / denom1) / (Sqrt(_3 / denom2) + eps_),
          params,
          mt_,
          vt_);

 cudaStreamSynchronize(0);
}

void Adam::updateState(Ptr<OptimizerBase> localOpt, size_t shardSize_, int my_id) {
  if(!mtAlloc_ || !vtAlloc_) {
    return; //We don't update the optimizer unless it's initialized
  }
  Tensor remoteMT = localOpt->getMT_();
  Tensor remoteVT = localOpt->getVT_();
  int pos = shardSize_*my_id;
  mt_->copyFrom(remoteMT->subtensor(pos, mt_->size()));
  vt_->copyFrom(remoteVT->subtensor(pos, vt_->size()));
}

Ptr<OptimizerBase> Optimizer(Ptr<Config> options) {
  Ptr<ClipperBase> clipper = nullptr;
  float clipNorm = options->get<double>("clip-norm");
  if(clipNorm > 0)
    clipper = Clipper<Norm>(clipNorm);

  float lrate = options->get<double>("learn-rate");

  std::string opt = options->get<std::string>("optimizer");

  float beta1 = options->get<double>("beta1");
  float beta2 = options->get<double>("beta2");
  float eps = options->get<double>("eps");

  Ptr<OptimizerBase> ret;
  if(opt == "sgd") {
    ret = Optimizer<Sgd>(lrate, keywords::clip = clipper);
  } else if(opt == "adagrad") {
    ret = Optimizer<Adagrad>(lrate, keywords::clip = clipper);
  } else if(opt == "adam") {
    ret = Optimizer<Adam>(lrate, keywords::clip = clipper);
  } else {
    UTIL_THROW2("Unknown optimizer: " << opt);
  }

  ret->setB1(beta1);
  ret->setB2(beta2);
  ret->setEPS(eps);
  return ret;
}

Ptr<OptimizerBase> Optimizer(std::string opt, double lrate, double clipNorm) {
  Ptr<ClipperBase> clipper = nullptr;
  if(clipNorm > 0)
    clipper = Clipper<Norm>(clipNorm);

  if(opt == "sgd") {
    return Optimizer<Sgd>(lrate, keywords::clip = clipper);
  } else if(opt == "adagrad") {
    return Optimizer<Adagrad>(lrate, keywords::clip = clipper);
  } else if(opt == "adam") {
    return Optimizer<Adam>(lrate, keywords::clip = clipper);
  } else {
    UTIL_THROW2("Unknown optimizer: " << opt);
  }
}

}
