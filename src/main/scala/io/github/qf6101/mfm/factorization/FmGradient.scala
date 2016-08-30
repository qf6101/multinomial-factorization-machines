package io.github.qf6101.mfm.factorization

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.base.Coefficients
import io.github.qf6101.mfm.optimization.Gradient
import io.github.qf6101.mfm.util.Logging
import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 15-3-11.
  */
class FmGradient(paramHolder: FmModelParam, paramPool: ParamMap) extends Gradient with Logging {
  /**
    * Compute the gradient and loss given the features of a single data point,
    * add the gradient to a provided vector to avoid creating new objects, and return loss.
    *
    * @param data features for one data point
    * @param label label for this data point
    * @param coeffs weights/coefficients corresponding to features
    * @param cumGradient the computed gradient will be added to this vector
    * @param negativePenalty 不平衡数据中负样本的惩罚比例
    * @return loss
    */
  override def compute(data: SparseVector[Double],
                       label: Double,
                       coeffs: Coefficients,
                       cumGradient: Coefficients,
                       negativePenalty: Double):
  Double = {
    val fmcoeffs = coeffs.asInstanceOf[FmCoefficients]
    val fmCumGradient = cumGradient.asInstanceOf[FmCoefficients]
    val linearScore = FmModel.linearScore(data, paramHolder, paramPool, fmcoeffs)
    val expComponent = 1 + math.exp(-label * linearScore)
    val loss = math.log(expComponent)
    val multiplier = -label * (1 - 1 / expComponent)
    //参与2阶项的最大维度
    val maxInteractAttr = paramPool(paramHolder.maxInteractAttr)
    //0阶梯度
    if (paramPool(paramHolder.k0)) {
      val incrementGradient = if(label > 0) multiplier else multiplier * negativePenalty
      fmCumGradient.w0 += incrementGradient
    }
    //1阶梯度
    if (paramPool(paramHolder.k1)) {
      data.activeIterator.foreach { case (index, value) =>
        val incrementGradient = if(label > 0) multiplier * value else multiplier * value * negativePenalty
        fmCumGradient.w(index) += incrementGradient
      }
    }
    //2阶梯度
    if (paramPool(paramHolder.k2)) {
      for (factorIndex <- 0 until paramPool(paramHolder.numFactors)) {
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
            fmCumGradient.v.update(index, factorIndex, if(label > 0) incrementGradient else incrementGradient * negativePenalty)
          }
        }
      }
    }
    loss
  }

}
