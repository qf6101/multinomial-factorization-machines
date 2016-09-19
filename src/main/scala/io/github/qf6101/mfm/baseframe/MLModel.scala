package io.github.qf6101.mfm.baseframe

import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 16-9-8.
  */

/**
  * 机器学习模型基类
  *
  * @param paramMeta 模型参赛
  * @param coeffs    模型系数
  * @param params    参数池（保存参数的值）
  */
abstract class MLModel(val paramMeta: ModelParam,
                       val coeffs: Coefficients,
                       val params: ParamMap) extends Serializable {
  /**
    * 保存模型文件
    *
    * @param location 模型文件的位置
    */
  def save(location: String): Unit = {
    //保存模型系数
    coeffs.save(location + "/" + MLModel.namingCoeffFile)
    //保存模型参数
    paramMeta.save(location + "/" + MLModel.namingParamFile, params)
  }

  /**
    * 模型内容是否相同
    *
    * @param other 另一个模型
    * @return 内容是否相同
    */
  def equals(other: MLModel): Boolean
}

/**
  * 静态模型对象
  */
object MLModel {
  val namingCoeffFile = "coefficient"
  val namingParamFile = "params"
}