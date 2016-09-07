package io.github.qf6101.mfm.factorization

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.base.Coefficients
import io.github.qf6101.mfm.optimization.Gradient
import io.github.qf6101.mfm.util.Logging

/**
  * Created by qfeng on 16-9-7.
  */
class MfmGradient extends Gradient with Logging {
  /**
    * Compute the gradient and loss given the features of a single data point,
    * add the gradient to a provided vector to avoid creating new objects, and return loss.
    *
    * @param data        features for one data point
    * @param label       label for this data point
    * @param coeffs      weights/coefficients corresponding to features
    * @param cumGradient the computed gradient will be added to this vector
    * @return loss
    */
  override def compute(data: SparseVector[Double], label: Double, coeffs: Coefficients, cumGradient: Coefficients): Double = ???
}
