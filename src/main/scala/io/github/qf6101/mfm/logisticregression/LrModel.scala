package io.github.qf6101.mfm.logisticregression

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.MLModel
import io.github.qf6101.mfm.baseframe.binomial.BinModel
import io.github.qf6101.mfm.util.Logging
import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 15-3-16.
  */

/**
  * 逻辑斯蒂回归模型
  *
  * @param coeffs    模型系数
  * @param paramMeta 逻辑斯蒂参数
  * @param params    参数池
  */
class LrModel(override val paramMeta: LrModelParam,
              override val coeffs: VectorCoefficients,
              override val params: ParamMap)
  extends BinModel(paramMeta, coeffs, params) with Logging {
  /**
    * 对输入数据进行预测（使用内置系数）
    *
    * @param data 输入数据
    * @return 预测值(0~1)
    */
  override def predict(data: SparseVector[Double]): Double = {
    predict(data, this.coeffs)
  }

  /**
    * 对输入数据进行预测
    *
    * @param data   输入数据
    * @param coeffs 系数
    * @return 预测值(0~1)
    */
  def predict(data: SparseVector[Double], coeffs: VectorCoefficients = this.coeffs): Double = {
    val margin = -1.0 * coeffs.dot(data)
    1.0 / (1.0 + math.exp(margin))
  }

  override def equals(other: MLModel): Boolean = {
    other match {
      case otherModel: LrModel =>
        if (paramMeta.toJSON(params).equals(otherModel.paramMeta.toJSON(otherModel.params))
          && coeffs.equals(otherModel.coeffs)) true
        else false
      case _ => false
    }
  }
}

object LrModel extends Logging {
  /**
    * 从模型文件载入逻辑斯蒂模型
    *
    * @param location 模型文件
    * @return 逻辑斯蒂模型
    */
  def apply(location: String): LrModel = {
    val params = new ParamMap()
    val paramMeta = LrModelParam(location + "/" + MLModel.namingParamFile, params)
    val coefficients = VectorCoefficients(location + "/" + MLModel.namingCoeffFile)
    new LrModel(paramMeta, coefficients, params)
  }
}