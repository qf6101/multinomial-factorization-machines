package io.github.qf6101.mfm.baseframe.binomial

import io.github.qf6101.mfm.baseframe.ModelParam
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.json4s.JsonAST
import org.json4s.JsonDSL._

/**
  * Created by qfeng on 16-9-8.
  */
trait BinModelParam extends ModelParam {
  //default value: 0.5
  val binaryThreshold: Param[Double] = new Param("ModelParam", "binaryThreshold", "threshold for binary classification", ParamValidators.inRange(0, 1, false, false))

  /**
    * Transform parameters to json object
    *
    * @return parameters in json format
    */
  override def toJSON(params: ParamMap): JsonAST.JObject = {
    super.toJSON(params) ~ (binaryThreshold.name -> params(binaryThreshold))
  }
}
