package io.github.qf6101.mfm.factorization.multinomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.mutinomial.MultiModel
import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 16-9-7.
  */
class MfmModel(override val coeffs: MfmCoefficients,
               override val paramHolder: MfmModelParam,
               override val paramPool: ParamMap) extends MultiModel(coeffs, paramHolder, paramPool) {
  /**
    * 对输入数据进行预测
    *
    * @param data 输入数据
    * @return 预测值(0~1)
    */
  override def regressionPredict(data: SparseVector[Double]): Array[Double] = ???
}
