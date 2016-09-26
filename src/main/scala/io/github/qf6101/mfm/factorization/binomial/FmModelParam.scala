package io.github.qf6101.mfm.factorization.binomial

import better.files.File
import io.github.qf6101.mfm.baseframe.ModelParam
import io.github.qf6101.mfm.baseframe.binomial.BinModelParam
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.sql.SparkSession
import org.json4s.{DefaultFormats, JsonAST}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

/**
  * Created by qfeng on 15-1-26.
  */

/**
  * Factorization Machine模型参数
  */
trait FmModelParam extends BinModelParam {
  val numFeatures: Param[Int] = new Param("FmModelParam", "numFeatures", "样本维度数", ParamValidators.gt(0))
  val numFactors: Param[Int] = new Param("FmModelParam", "numFactors", "2阶分解维度数", ParamValidators.gt(0))
  val k0: Param[Boolean] = new Param("FmModelParam", "k0", "是否考虑0阶", ParamValidators.inArray(Array(true, false)))
  val k1: Param[Boolean] = new Param("FmModelParam", "k1", "是否考虑1阶", ParamValidators.inArray(Array(true, false)))
  val k2: Param[Boolean] = new Param("FmModelParam", "k2", "是否考虑2阶", ParamValidators.inArray(Array(true, false)))
  val reg0: Param[Double] = new Param("FmModelParam", "reg0", "正则参数")
  val reg1: Param[Double] = new Param("FmModelParam", "reg1", "正则参数")
  val reg2: Param[Double] = new Param("FmModelParam", "reg2", "正则参数")
  val maxInteractFeatures: Param[Int] = new Param("FmModelParam", "maxInteractFeatures", "参与2阶项的最大特征维度（不包含）", ParamValidators.gt(0))

  /**
    * Transform parameters to json object
    *
    * @return parameters in json format
    */
  override def toJSON(params: ParamMap): JsonAST.JObject = {
    super.toJSON(params) ~
      (ModelParam.namingParamType -> FmModelParam.getClass.toString) ~
      (reg0.name -> params(reg0)) ~
      (reg1.name -> params(reg1)) ~
      (reg2.name -> params(reg2)) ~
      (numFeatures.name -> params(numFeatures)) ~
      (numFactors.name -> params(numFactors)) ~
      (k0.name -> params(k0)) ~
      (k1.name -> params(k1)) ~
      (k2.name -> params(k2)) ~
      (maxInteractFeatures.name -> params(maxInteractFeatures))
  }
}

object FmModelParam {
  /**
    * 从参数文件构造分解机模型参数
    *
    * @param location 参数文件位置
    * @param params   参数池
    * @return 分解机型参数
    */
  def apply(location: String, params: ParamMap): FmModelParam = {
    // 初始化参数对象和spark session
    val fmModelParam = new FmModelParam {}
    val spark = SparkSession.builder().getOrCreate()
    // 读取参数值
    val paramValues = spark.read.json(location).first()
    val binaryThreshold = paramValues.getAs[Double](fmModelParam.binaryThreshold.name)
    val reg0 = paramValues.getAs[Double](fmModelParam.reg0.name)
    val reg1 = paramValues.getAs[Double](fmModelParam.reg1.name)
    val reg2 = paramValues.getAs[Double](fmModelParam.reg2.name)
    val numFeatures = paramValues.getAs[Long](fmModelParam.numFeatures.name).toInt
    val numFactors = paramValues.getAs[Long](fmModelParam.numFactors.name).toInt
    val k0 = paramValues.getAs[Boolean](fmModelParam.k0.name)
    val k1 = paramValues.getAs[Boolean](fmModelParam.k1.name)
    val k2 = paramValues.getAs[Boolean](fmModelParam.k2.name)
    val initMean = paramValues.getAs[Double](fmModelParam.initMean.name)
    val initStdev = paramValues.getAs[Double](fmModelParam.initStdev.name)
    val maxInteractFeatures = paramValues.getAs[Long](fmModelParam.maxInteractFeatures.name).toInt
    // 设置参数值
    params.put(fmModelParam.binaryThreshold, binaryThreshold)
    params.put(fmModelParam.reg0, reg0)
    params.put(fmModelParam.reg1, reg1)
    params.put(fmModelParam.reg2, reg2)
    params.put(fmModelParam.numFeatures, numFeatures)
    params.put(fmModelParam.numFactors, numFactors)
    params.put(fmModelParam.k0, k0)
    params.put(fmModelParam.k1, k1)
    params.put(fmModelParam.k2, k2)
    params.put(fmModelParam.initMean, initMean)
    params.put(fmModelParam.initStdev, initStdev)
    params.put(fmModelParam.maxInteractFeatures, maxInteractFeatures)
    // 返回FM参数
    fmModelParam
  }

  /**
    * 从本地文件载入参数
    *
    * @param location 本地文件位置
    * @param params 参数池
    * @return 分解机参数
    */
  def fromLocal(location: String, params: ParamMap): FmModelParam = {
    // 初始化参数对象
    val fmModelParam = new FmModelParam {}
    implicit val formats = DefaultFormats
    // 读取参数值
    val paramValues = parse(File(location).contentAsString)
    val binaryThreshold = (paramValues \ fmModelParam.binaryThreshold.name).extract[Double]
    val reg0 = (paramValues \ fmModelParam.reg0.name).extract[Double]
    val reg1 = (paramValues \ fmModelParam.reg1.name).extract[Double]
    val reg2 = (paramValues \ fmModelParam.reg2.name).extract[Double]
    val numFeatures = (paramValues \ fmModelParam.numFeatures.name).extract[Int]
    val numFactors = (paramValues \ fmModelParam.numFactors.name).extract[Int]
    val k0 = (paramValues \ fmModelParam.k0.name).extract[Boolean]
    val k1 = (paramValues \ fmModelParam.k1.name).extract[Boolean]
    val k2 = (paramValues \ fmModelParam.k2.name).extract[Boolean]
    val initMean = (paramValues \ fmModelParam.initMean.name).extract[Double]
    val initStdev = (paramValues \ fmModelParam.initStdev.name).extract[Double]
    val maxInteractFeatures = (paramValues \ fmModelParam.maxInteractFeatures.name).extract[Int]
    // 设置参数值
    params.put(fmModelParam.binaryThreshold, binaryThreshold)
    params.put(fmModelParam.reg0, reg0)
    params.put(fmModelParam.reg1, reg1)
    params.put(fmModelParam.reg2, reg2)
    params.put(fmModelParam.numFeatures, numFeatures)
    params.put(fmModelParam.numFactors, numFactors)
    params.put(fmModelParam.k0, k0)
    params.put(fmModelParam.k1, k1)
    params.put(fmModelParam.k2, k2)
    params.put(fmModelParam.initMean, initMean)
    params.put(fmModelParam.initStdev, initStdev)
    params.put(fmModelParam.maxInteractFeatures, maxInteractFeatures)
    // 返回FM参数
    fmModelParam
  }
}