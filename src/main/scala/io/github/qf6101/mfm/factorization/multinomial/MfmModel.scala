package io.github.qf6101.mfm.factorization.multinomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.MLModel
import io.github.qf6101.mfm.baseframe.mutinomial.MultiModel
import io.github.qf6101.mfm.factorization.binomial.FmModel
import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 16-9-7.
  */

/**
  * 多分类FM模型
  *
  * @param paramMeta 模型参赛
  * @param coeffs    模型系数
  * @param params    参数池（保存参数的值）
  */
class MfmModel(override val paramMeta: MfmModelParam,
               override val coeffs: MfmCoefficients,
               override val params: ParamMap) extends MultiModel(paramMeta, coeffs, params) {
  //dump, 设置默认的阈值为0.5
  params.put(paramMeta.binaryThreshold, 0.5)

  /**
    * 对输入数据进行预测
    *
    * @param data 输入数据
    * @return 预测值向量(0~1)
    */
  override def predict(data: SparseVector[Double]): Array[Double] = {
    MfmModel.predict(data, paramMeta, params, coeffs)
  }

  /**
    * 模型内容是否相同
    *
    * @param other 另一个模型
    * @return 内容是否相同
    */
  override def equals(other: MLModel): Boolean = {
    other match {
      case otherModel: MfmModel =>
        if (paramMeta.toJSON(params).equals(otherModel.paramMeta.toJSON(otherModel.params))
          && coeffs.equals(otherModel.coeffs)) true
        else false
      case _ => false
    }
  }
}

/**
  * 多分类FM模型对象
  */
object MfmModel {
  /**
    * 对输入样本进行预测
    *
    * @param data      样本数据
    * @param paramMeta 多分类FM参数
    * @param params    参数池
    * @param coeffs    多分类FM系数
    * @return 预测值
    */
  def predict(data: SparseVector[Double],
              paramMeta: MfmModelParam,
              params: ParamMap,
              coeffs: MfmCoefficients): Array[Double] = {
    // 计算线性得分
    val scores = coeffs.thetas.map { theta =>
      FmModel.linearScore(data, paramMeta, params, theta)
    }
    // 为了防止溢出,对分子分母都除以maxScore,得到adjustedScores
    val maxScore = scores.max
    val adjustedScores = scores.map { score =>
      math.exp(score - maxScore)
    }
    // 计算归一化后的得分
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