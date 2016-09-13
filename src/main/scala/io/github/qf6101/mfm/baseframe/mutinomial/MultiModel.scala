package io.github.qf6101.mfm.baseframe.mutinomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.{Coefficients, MLModel}
import io.github.qf6101.mfm.util.Logging
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD

/**
  * Created by qfeng on 16-9-9.
  */
abstract class MultiModel(override val paramMeta: MultiModelParam,
                          override val coeffs: Coefficients,
                          override val params: ParamMap)
  extends MLModel(paramMeta, coeffs, params) with Logging with Serializable {
  /**
    * 对输入数据进行预测
    *
    * @param data 输入数据
    * @return 预测值向量(0~1)
    */
  def predict(data: SparseVector[Double]): Array[Double]

  /**
    * 对输入数据集进行预测
    *
    * @param dataSet 输入数据集
    * @return 预测值集合(0~1)
    */
  def predict(dataSet: RDD[SparseVector[Double]]): RDD[Array[Double]] = {
    dataSet.map(predict)
  }
}