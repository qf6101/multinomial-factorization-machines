package io.github.qf6101.mfm.optimization

import io.github.qf6101.mfm.baseframe.Coefficients
import io.github.qf6101.mfm.util.Logging

/**
  * Created by qfeng on 15-3-11.
  */
abstract class Updater(private val decreasingStrategy: DecreasingStrategy)
  extends Logging with Serializable {
  /**
    * Compute an updated value for weights given the gradient, stepSize, iteration number and
    * regularization parameter. Also returns the regularization value regParam * R(w)
    * computed using the *updated* weights.
    *
    * @param coeffOld - Old coefficients.
    * @param gradient - Average batch gradient.
    * @param stepSize - step size across iterations
    * @param iter - Iteration number
    * @param regParam - Regularization parameter
    *
    * @return A tuple of 2 elements. The first element is a coefficient structure containing updated weights,
    *         and the second element is the regularization value computed using updated weights.
    */
  def compute(coeffOld: Coefficients,
              gradient: Coefficients,
              stepSize: Double,
              iter: Int,
              regParam: Array[Double]): (Coefficients, Double)
}

/**
  * A simple updater for gradient descent *without* any regularization.
  * Uses a step-size decreasing with the square root of the number of iterations.
  */
class SimpleUpdater(private val decreasingStrategy: DecreasingStrategy = new sqrtDecreasingStrategy())
extends Updater(decreasingStrategy) {
  /**
    * Compute an updated value for weights given the gradient, stepSize, iteration number and
    * regularization parameter. Also returns the regularization value regParam * R(w)
    * computed using the *updated* weights.
    *
    * @param coeffOld - Old coefficients.
    * @param gradient - Average batch gradient.
    * @param stepSize - step size across iterations
    * @param iter - Iteration number
    * @param regParam - Regularization parameter
    *
    * @return A tuple of 2 elements. The first element is a coefficient structure containing updated weights,
    *         and the second element is the regularization value computed using updated weights.
    */
  override def compute(coeffOld: Coefficients,
                       gradient: Coefficients,
                       stepSize: Double,
                       iter: Int,
                       regParam: Array[Double]): (Coefficients, Double) = {
    val thisIterStepSize = stepSize / decreasingStrategy.decrease(iter)
    val coeffNew = coeffOld + gradient * (-thisIterStepSize)
    (coeffNew, 0.0)
  }
}

/**
  * Updater for L2 regularized problems.
  * R(w) = 1/2 ||w||^2
  * Uses a step-size decreasing with the square root of the number of iterations.
  */
class SquaredL2Updater(private val decreasingStrategy: DecreasingStrategy = new sqrtDecreasingStrategy())
  extends Updater(decreasingStrategy) {
  /**
    * Compute an updated value for weights given the gradient, stepSize, iteration number and
    * regularization parameter. Also returns the regularization value regParam * R(w)
    * computed using the *updated* weights.
    *
    * @param coeffOld - Old coefficients.
    * @param gradient - Average batch gradient.
    * @param stepSize - step size across iterations
    * @param iter - Iteration number
    * @param regParam - Regularization parameter
    *
    * @return A tuple of 2 elements. The first element is a coefficient structure containing updated weights,
    *         and the second element is the regularization value computed using updated weights.
    */
  override def compute(coeffOld: Coefficients,
                       gradient: Coefficients,
                       stepSize: Double,
                       iter: Int,
                       regParam: Array[Double]): (Coefficients, Double) = {
    // add up both updates from the gradient of the loss (= step) as well as
    // the gradient of the regularizer (= regParam * weightsOld)
    // w' = w - thisIterStepSize * (gradient + regParam * w)
    val thisIterStepSize = stepSize / decreasingStrategy.decrease(iter)
    val coeffNew = coeffOld + (((gradient + coeffOld.L2RegGradient(regParam)) * (-thisIterStepSize)))
    (coeffNew, coeffNew.L2RegValue(regParam))
  }
}

/**
  * :: DeveloperApi ::
  * Updater for L1 regularized problems.
  * R(w) = ||w||_1
  * Uses a step-size decreasing with the square root of the number of iterations.

  * Instead of subgradient of the regularizer, the proximal operator for the
  * L1 regularization is applied after the gradient step. This is known to
  * result in better sparsity of the intermediate solution.
  *
  * The corresponding proximal operator for the L1 norm is the soft-thresholding
  * function. That is, each weight component is shrunk towards 0 by shrinkageVal.
  *
  * If w >  shrinkageVal, set weight component to w-shrinkageVal.
  * If w < -shrinkageVal, set weight component to w+shrinkageVal.
  * If -shrinkageVal < w < shrinkageVal, set weight component to 0.
  *
  * Equivalently, set weight component to signum(w) * max(0.0, abs(w) - shrinkageVal)
  */
class L1Updater(private val decreasingStrategy: DecreasingStrategy = new sqrtDecreasingStrategy())
  extends Updater(decreasingStrategy) {
  /**
    * Compute an updated value for weights given the gradient, stepSize, iteration number and
    * regularization parameter. Also returns the regularization value regParam * R(w)
    * computed using the *updated* weights.
    *
    * @param coeffOld - Old coefficients.
    * @param gradient - Average batch gradient.
    * @param stepSize - step size across iterations
    * @param iter - Iteration number
    * @param regParam - Regularization parameter
    *
    * @return A tuple of 2 elements. The first element is a coefficient structure containing updated weights,
    *         and the second element is the regularization value computed using updated weights.
    */
  override def compute(coeffOld: Coefficients,
                       gradient: Coefficients,
                       stepSize: Double,
                       iter: Int,
                       regParam: Array[Double]): (Coefficients, Double) = {
    val thisIterStepSize = stepSize / decreasingStrategy.decrease(iter)
    val coeffNew = coeffOld + (gradient * (-thisIterStepSize))
    // Apply proximal operator (soft thresholding)
    coeffNew.L1Shrink(regParam, thisIterStepSize)
    (coeffNew, coeffNew.L1RegValue(regParam))
  }
}
