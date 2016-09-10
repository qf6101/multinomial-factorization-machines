package io.github.qf6101.mfm.factorization.binary

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.Coefficients
import io.github.qf6101.mfm.optimization.Gradient
import io.github.qf6101.mfm.util.Logging
import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 15-3-11.
  */
class FmGradient(paramMeta: FmModelParam, params: ParamMap) extends Gradient with Logging {
  /**
    * Compute the gradient and loss given the features of a single data point,
    * add the gradient to a provided vector to avoid creating new objects, and return loss.
    *
    * @param data features for one data point
    * @param label label for this data point
    * @param coeffs weights/coefficients corresponding to features
    * @param cumGradient the computed gradient will be added to this vector
    * @return loss
    */
  override def compute(data: SparseVector[Double],
                       label: Double,
                       coeffs: Coefficients,
                       cumGradient: Coefficients):
  Double = {
    val fmcoeffs = coeffs.asInstanceOf[FmCoefficients]
    val fmCumGradient = cumGradient.asInstanceOf[FmCoefficients]
    val linearScore = FmModel.linearScore(data, paramMeta, params, fmcoeffs)
    val expComponent = 1 + math.exp(-label * linearScore)
    val loss = math.log(expComponent)
    val multiplier = -label * (1 - 1 / expComponent)
    //参与2阶项的最大维度
    val maxInteractAttr = params(paramMeta.maxInteractFeatures)
    //0阶梯度
    if (params(paramMeta.k0)) {
      fmCumGradient.w0 += multiplier
    }
    //1阶梯度
    if (params(paramMeta.k1)) {
      data.activeIterator.foreach { case (index, value) =>
        fmCumGradient.w(index) += multiplier * value
      }
    }
    //2阶梯度
    if (params(paramMeta.k2)) {
      for (factorIndex <- 0 until params(paramMeta.numFactors)) {
        //提前计算（因为求和中每一项都会用到）firstMoment = \sum_j^n {v_jf*x_j} （固定f）
        val firstMoment = data.activeIterator.foldLeft(0.0) { case (sum, (index, value)) =>
          if (index < maxInteractAttr) {
            sum + fmcoeffs.v(index, factorIndex) * value
          } else sum
        }
        //计算2阶梯度
        data.activeIterator.foreach { case (index, value) =>
          if (index < maxInteractAttr) {
            val twoWayCumCoeff = fmCumGradient.v(index, factorIndex)
            val twoWayCoeff = fmcoeffs.v(index, factorIndex)
            val incrementGradient = twoWayCumCoeff + multiplier * ((value * firstMoment) - (twoWayCoeff * value * value))
            fmCumGradient.v.update(index, factorIndex, incrementGradient)
          }
        }
      }
    }
    loss
  }

}
