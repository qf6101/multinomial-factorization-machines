package io.github.qf6101.mfm.baseframe.binary

import io.github.qf6101.mfm.baseframe.ModelParam
import org.apache.spark.ml.param.{Param, ParamValidators}

/**
  * Created by qfeng on 16-9-8.
  */
trait BinModelParam extends ModelParam {
  //default value: 0.5
  val binaryThreshold: Param[Double] = new Param("ModelParam", "binaryThreshold", "threshold for binary classification", ParamValidators.inRange(0, 1, false, false))
}
