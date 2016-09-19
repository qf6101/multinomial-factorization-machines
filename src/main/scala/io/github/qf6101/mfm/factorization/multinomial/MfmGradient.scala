package io.github.qf6101.mfm.factorization.multinomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.Coefficients
import io.github.qf6101.mfm.optimization.Gradient
import io.github.qf6101.mfm.util.Logging
import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 16-9-7.
  */

/**
  * 多分类FM梯度
  *
  * @param paramMeta 多分类FM参数
  * @param params    参数池
  */
class MfmGradient(paramMeta: MfmModelParam, params: ParamMap) extends Gradient with Logging {
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
                       cumGradient: Coefficients): Double = {
    val mfmCoeff = coeffs.asInstanceOf[MfmCoefficients]
    val mfmCumGradient = cumGradient.asInstanceOf[MfmCoefficients]
    val scores = MfmModel.predict(data, paramMeta, params, mfmCoeff)
    val multipliers = scores.zipWithIndex.map { case (score, index) =>
      if (label.toInt == index) score - 1.0 else score
    }
    //参与2阶项的最大维度
    val maxInteractFeatures = params(paramMeta.maxInteractFeatures)
    val loss = -math.log(scores(label.toInt))
    (mfmCoeff.thetas zip mfmCumGradient.thetas zip multipliers).foreach { case ((fmCoeff, fmCumGradient), multiplier) =>
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
            if (index < maxInteractFeatures) {
              sum + fmCoeff.v(index, factorIndex) * value
            } else sum
          }
          //计算2阶梯度
          data.activeIterator.foreach { case (index, value) =>
            if (index < maxInteractFeatures) {
              val twoWayCumCoeff = fmCumGradient.v(index, factorIndex)
              val twoWayCoeff = fmCoeff.v(index, factorIndex)
              val incrementGradient = twoWayCumCoeff + multiplier * ((value * firstMoment) - (twoWayCoeff * value * value))
              fmCumGradient.v.update(index, factorIndex, incrementGradient)
            }
          }
        }
      }
    }
    loss
  }
}
