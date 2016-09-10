package io.github.qf6101.mfm.baseframe.mutinomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.{Coefficients, MLModel}
import io.github.qf6101.mfm.util.Logging
import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 16-9-9.
  */
abstract class MultiModel(override val coeffs: Coefficients,
                          val paramMeta: MultiModelParam,
                          override val params: ParamMap) extends MLModel(coeffs, params) with Logging with Serializable {
  /**
    * 对输入数据进行预测
    *
    * @param data 输入数据
    * @return 预测值向量(0~1)
    */
  def regressionPredict(data: SparseVector[Double]): Array[Double]
}