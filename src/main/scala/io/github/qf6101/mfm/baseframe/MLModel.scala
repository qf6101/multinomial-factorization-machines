package io.github.qf6101.mfm.baseframe

import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 16-9-8.
  */
abstract class MLModel(val coeffs: Coefficients,
                       val paramPool: ParamMap) extends Serializable
