package io.github.qf6101.mfm.factorization.binomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.MLModel
import io.github.qf6101.mfm.baseframe.binomial.BinModel
import io.github.qf6101.mfm.util.Logging
import org.apache.spark.ml.param.ParamMap


/**
  * Created by qfeng on 15-1-26.
  */

/**
  * Factorization Machine模型
  *
  * @param paramMeta 分解机模型参数
  * @param params    参数池
  */
class FmModel(override val paramMeta: FmModelParam,
              override val coeffs: FmCoefficients,
              override val params: ParamMap)
  extends BinModel(paramMeta, coeffs, params) {
  /**
    * 对输入数据进行预测
    *
    * @param data 输入数据
    * @return 预测值
    */
  override def predict(data: SparseVector[Double]): Double = {
    val score = FmModel.linearScore(data, paramMeta, params, coeffs)
    1.0 / (1.0 + math.exp(-score))
  }

  /**
    * 模型内容是否相同
    *
    * @param other 另一个模型
    * @return 内容是否相同
    */
  override def equals(other: MLModel): Boolean = {
    other match {
      case otherModel: FmModel =>
        if (paramMeta.toJSON(params).equals(otherModel.paramMeta.toJSON(otherModel.params))
          && coeffs.equals(otherModel.coeffs)) true
        else false
      case _ => false
    }
  }
}

object FmModel extends Logging {
  /**
    * 计算样本的线性得分值
    *
    * @param data      样本
    * @param paramMeta 参数元数据
    * @param params    参数池
    * @param coeffs    FM系数
    * @return 输入样本的线性得分值
    */
  def linearScore(data: SparseVector[Double],
                  paramMeta: FmModelParam,
                  params: ParamMap,
                  coeffs: FmCoefficients): Double = {
    //初始化各阶预测值为0
    var zeroWayPredict = 0.0
    var oneWayPredict = 0.0
    var twoWayPredict = 0.0
    //参与2阶项的最大维度
    val maxInteractAttr = params(paramMeta.maxInteractFeatures)
    //0阶预测值
    if (params(paramMeta.k0)) {
      zeroWayPredict += coeffs.w0
    }
    //1阶预测值
    if (params(paramMeta.k1)) {
      data.activeIterator.foreach { case (index, value) =>
        oneWayPredict += coeffs.w(index) * value
      }
    }
    //2阶预测值
    if (params(paramMeta.k2)) {
      for (factorIndex <- 0 until params(paramMeta.numFactors)) {
        var firstMoment = 0.0
        var secondMoment = 0.0
        data.activeIterator.foreach { case (index, value) =>
          if (index < maxInteractAttr) {
            firstMoment += coeffs.v(index, factorIndex) * value
            secondMoment += math.pow(coeffs.v(index, factorIndex) * value, 2)
          }
        }
        twoWayPredict += firstMoment * firstMoment - secondMoment
      }
    }
    zeroWayPredict + oneWayPredict + 0.5 * twoWayPredict
  }

  /**
    * 从文件载入分解机模型
    *
    * @param location 包含分解机型信息的文件
    * @return 分解机模型
    */
  def apply(location: String): FmModel = {
    val params = new ParamMap()
    val paramMeta = FmModelParam(location + "/" + MLModel.namingParamFile, params)
    val coefficients = FmCoefficients(location + "/" + MLModel.namingCoeffFile)
    new FmModel(paramMeta, coefficients, params)
  }

  /**
    * 从本地文件载入分解机模型
    *
    * @param location 包含分解机型信息的本地文件
    * @return 分解机模型
    */
  def fromLocal(location: String): FmModel = {
    val params = new ParamMap()
    val paramMeta = FmModelParam.fromLocal(location + "/" + MLModel.namingParamFile + "/part-00000", params)
    val coefficients = FmCoefficients.fromLocal(location + "/" + MLModel.namingCoeffFile)
    new FmModel(paramMeta, coefficients, params)
  }
}