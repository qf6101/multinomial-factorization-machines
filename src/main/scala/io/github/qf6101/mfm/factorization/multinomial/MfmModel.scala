package io.github.qf6101.mfm.factorization.multinomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.MLModel
import io.github.qf6101.mfm.baseframe.mutinomial.MultiModel
import io.github.qf6101.mfm.factorization.binomial.{FmCoefficients, FmModel, FmModelParam}
import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 16-9-7.
  */
class MfmModel(override val paramMeta: MfmModelParam,
               override val coeffs: MfmCoefficients,
               override val params: ParamMap) extends MultiModel(paramMeta, coeffs, params) {
  /**
    * 对输入数据进行预测
    *
    * @param data 输入数据
    * @return 预测值向量(0~1)
    */
  override def predict(data: SparseVector[Double]): Array[Double] = {
    MfmModel.predict(data, paramMeta, params, coeffs)
  }
}

object MfmModel {
  def predict(data: SparseVector[Double],
              paramMeta: MfmModelParam,
              params: ParamMap,
              coeffs: MfmCoefficients): Array[Double] = {
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

  /**
    * 从文件载入分解机模型
    *
    * @param location 包含分解机型信息的文件
    * @return 分解机模型
    */
  def apply(location: String): MfmModel = {
    val params = new ParamMap()
    val paramMeta = MfmModelParam(location + "/" + MLModel.namingParamFile, params)
    val coefficients = MfmCoefficients(location + "/" + MLModel.namingCoeffFile)
    new MfmModel(paramMeta, coefficients, params)
  }
}