package io.github.qf6101.mfm.baseframe

import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 16-9-8.
  */
abstract class MLModel(val paramMeta: ModelParam,
                       val coeffs: Coefficients,
                       val params: ParamMap) extends Serializable {
  def save(location: String): Unit = {
    coeffs.save(location + "/" + MLModel.namingCoeffFile)
    paramMeta.save(location + "/" + MLModel.namingParamFile, params)
  }
}

object MLModel {
  val namingCoeffFile = "coefficient"
  val namingParamFile = "params"
}