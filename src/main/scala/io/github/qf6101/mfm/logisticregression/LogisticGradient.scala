package io.github.qf6101.mfm.logisticregression

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.Coefficients
import io.github.qf6101.mfm.optimization.Gradient
import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 15-3-13.
  */


/**
  * Compute gradient and loss for a logistic loss function, as used in binary classification.
  * See also the documentation for the precise formulation.
  */
class LogisticGradient(paramPool: ParamMap) extends Gradient {
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
  override def compute(data: SparseVector[Double],
                       label: Double,
                       coeffs: Coefficients,
                       cumGradient: Coefficients):
  Double = {
    val vecCoeffs = coeffs.asInstanceOf[VectorCoefficients]
    val vecCumGradient = cumGradient.asInstanceOf[VectorCoefficients]
    val hypotheses = 1 / (1 + math.exp(-1.0 * vecCoeffs.dot(data)))
    val multiplier = hypotheses - label

    vecCumGradient +=(multiplier, data * multiplier)
    if (label > 0) -math.log(hypotheses) else -math.log(1 - hypotheses)
  }
}
