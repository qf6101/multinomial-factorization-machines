package io.github.qf6101.mfm.optimization

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.base.Coefficients


/**
  * Created by qfeng on 15-3-11.
  */
trait Gradient extends Serializable {
  /**
    * Compute the gradient and loss given the features of a single data point.
    *
    * @param data features for one data point
    * @param label label for this data point
    * @param coeffs weights/coefficients corresponding to features
    *
    * @return (gradient: Coefficients, loss: Double)
    */
  def compute(data: SparseVector[Double],
              label: Double,
              coeffs: Coefficients): (Coefficients, Double) = {
    val gradient = coeffs.copyEmpty()
    val loss = compute(data, label, coeffs, gradient)
    (gradient, loss)
  }

  /**
    * Compute the gradient and loss given the features of a single data point,
    * add the gradient to a provided vector to avoid creating new objects, and return loss.
    *
    * @param data features for one data point
    * @param label label for this data point
    * @param coeffs weights/coefficients corresponding to features
    * @param cumGradient the computed gradient will be added to this vector
    *
    * @return loss
    */
  def compute(data: SparseVector[Double],
              label: Double,
              coeffs: Coefficients,
              cumGradient: Coefficients): Double
}
