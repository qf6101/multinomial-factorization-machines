package io.github.qf6101.mfm.logisticregression

import io.github.qf6101.mfm.baseframe.ModelParam
import io.github.qf6101.mfm.baseframe.binomial.BinModelParam
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.sql.SparkSession
import org.json4s.JsonAST
import org.json4s.JsonDSL._


/**
  * Created by qfeng on 15-3-18.
  */

/**
  * 逻辑斯蒂模型的参数
  */
trait LrModelParam extends BinModelParam {
  val reg: Param[Array[Double]] = new Param("LrModelParam", "reg", "正则参数")

  /**
    * Transform parameters to json object
    *
    * @return parameters in json format
    */
  override def toJSON(params: ParamMap): JsonAST.JObject = {
    super.toJSON(params) ~
      (ModelParam.namingParamType -> LrModelParam.getClass.toString) ~
      (reg.name -> params(reg).mkString(", "))
  }
}

object LrModelParam {
  /**
    * 根据字符串数组构造逻辑斯蒂模型参数
    *
    * @param location 文件位置
    * @param params   参数池
    * @return 逻辑斯蒂模型参数
    */
  def apply(location: String, params: ParamMap): LrModelParam = {
    val lrModelParam = new LrModelParam {}
    val spark = SparkSession.builder().getOrCreate()
    val paramValues = spark.read.json(location).first()
    val binaryThreshold = paramValues.getAs[Double](lrModelParam.binaryThreshold.name)
    val reg = paramValues.getAs[String](lrModelParam.reg.name).split(",").map(_.trim.toDouble)
    val initMean = paramValues.getAs[Double](lrModelParam.initMean.name)
    val initStdev = paramValues.getAs[Double](lrModelParam.initStdev.name)
    params.put(lrModelParam.binaryThreshold, binaryThreshold)
    params.put(lrModelParam.reg, reg)
    params.put(lrModelParam.initMean, initMean)
    params.put(lrModelParam.initStdev, initStdev)
    lrModelParam
  }
}
