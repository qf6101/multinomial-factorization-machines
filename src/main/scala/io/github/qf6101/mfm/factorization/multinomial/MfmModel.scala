package io.github.qf6101.mfm.factorization.multinomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.mutinomial.MultiModel
import io.github.qf6101.mfm.factorization.binomial.FmModel
import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 16-9-7.
  */
class MfmModel(override val coeffs: MfmCoefficients,
               override val paramMeta: MfmModelParam,
               override val params: ParamMap) extends MultiModel(coeffs, paramMeta, params) {
  /**
    * 对输入数据进行预测
    *
    * @param data 输入数据
    * @return 预测值向量(0~1)
    */
  override def regressionPredict(data: SparseVector[Double]): Array[Double] = {
    MfmModel.predict(data, paramMeta, params, coeffs)
  }
}

object MfmModel {
  def predict(data: SparseVector[Double],
              paramMeta: MfmModelParam,
              params: ParamMap,
              coeffs: MfmCoefficients):
  Array[Double] = {
    val scores = coeffs.thetas.map { theta =>
      FmModel.linearScore(data, paramMeta, params, theta)
    }
    val maxScore = scores.max
    val adjustedScores = scores.map { score =>
      math.exp(score - maxScore)
    }
    val sumAdjustedScores = adjustedScores.sum
    adjustedScores.map(_ / sumAdjustedScores)
  }
}