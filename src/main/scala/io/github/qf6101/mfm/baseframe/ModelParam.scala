package io.github.qf6101.mfm.baseframe

import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.sql.SparkSession
import org.json4s.JsonAST
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

/**
  * Created by qfeng on 15-4-2.
  */

/**
  * 模型参数
  */
trait ModelParam extends Serializable {
  val initMean: Param[Double] = new Param("ModelParam", "initMean", "使用高斯分布，初始化参数值，均值", ParamValidators.inRange(0, 1))
  val initStdev: Param[Double] = new Param("ModelParam", "initStdev", "使用高斯分布，初始化参数值，标准差值", ParamValidators.inRange(0, 1))


  /**
    * 将模型参数值保存至文件
    *
    * @param location 保存位置
    * @param params   参数池
    */
  def save(location: String, params: ParamMap): Unit = {
    SparkSession.builder().getOrCreate().sparkContext.
      makeRDD(List(compact(render(this.toJSON(params))))).repartition(1).saveAsTextFile(location)
  }

  /**
    * Transform parameters to json object
    *
    * @return parameters in json format
    */
  def toJSON(params: ParamMap): JsonAST.JObject = {
    (initMean.name -> params(initMean)) ~ (initStdev.name -> params(initStdev))
  }
}

/**
  * 静态模型参数对象
  */
object ModelParam {
  val namingParamType = "param_type"
}
