package io.github.qf6101.mfm.factorization.multinomial

import better.files.File
import io.github.qf6101.mfm.baseframe.ModelParam
import io.github.qf6101.mfm.baseframe.mutinomial.MultiModelParam
import io.github.qf6101.mfm.factorization.binomial.FmModelParam
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.sql.SparkSession
import org.json4s.JsonAST.JField
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.json4s.{DefaultFormats, JObject, JsonAST}


/**
  * Created by qfeng on 16-9-7.
  */

/**
  * 多分类FM模型参数
  */
trait MfmModelParam extends FmModelParam with MultiModelParam {
  val numClasses: Param[Int] = new Param("MfmModelParam", "numClasses", "标签数目", ParamValidators.gt(0))

  /**
    * Transform parameters to json object
    *
    * @return parameters in json format
    */
  override def toJSON(params: ParamMap): JsonAST.JObject = {
    val json = super.toJSON(params) removeField {
      case JField(ModelParam.namingParamType, _) => true
      case _ => false
    }
    json.asInstanceOf[JObject] ~
      (numClasses.name -> params(numClasses)) ~
      (ModelParam.namingParamType -> MfmModelParam.getClass.toString)
  }
}

object MfmModelParam {
  /**
    * 参数文件构造分解机模型参数
    *
    * @param location 文件位置
    * @param params   参数池
    * @return 分解机型参数
    */
  def apply(location: String, params: ParamMap): MfmModelParam = {
    // 初始化参数对象和spark session
    val mfmModelParam = new MfmModelParam {}
    val spark = SparkSession.builder().getOrCreate()
    // 读取参数值
    val paramValues = spark.read.json(location).first()
    val binaryThreshold = paramValues.getAs[Double](mfmModelParam.binaryThreshold.name)
    val reg0 = paramValues.getAs[Double](mfmModelParam.reg0.name)
    val reg1 = paramValues.getAs[Double](mfmModelParam.reg1.name)
    val reg2 = paramValues.getAs[Double](mfmModelParam.reg2.name)
    val numFeatures = paramValues.getAs[Long](mfmModelParam.numFeatures.name).toInt
    val numFactors = paramValues.getAs[Long](mfmModelParam.numFactors.name).toInt
    val k0 = paramValues.getAs[Boolean](mfmModelParam.k0.name)
    val k1 = paramValues.getAs[Boolean](mfmModelParam.k1.name)
    val k2 = paramValues.getAs[Boolean](mfmModelParam.k2.name)
    val initMean = paramValues.getAs[Double](mfmModelParam.initMean.name)
    val initStdev = paramValues.getAs[Double](mfmModelParam.initStdev.name)
    val maxInteractFeatures = paramValues.getAs[Long](mfmModelParam.maxInteractFeatures.name).toInt
    val numClasses = paramValues.getAs[Long](mfmModelParam.numClasses.name).toInt
    // 设置参数值
    params.put(mfmModelParam.binaryThreshold, binaryThreshold)
    params.put(mfmModelParam.reg0, reg0)
    params.put(mfmModelParam.reg1, reg1)
    params.put(mfmModelParam.reg2, reg2)
    params.put(mfmModelParam.numFeatures, numFeatures)
    params.put(mfmModelParam.numFactors, numFactors)
    params.put(mfmModelParam.k0, k0)
    params.put(mfmModelParam.k1, k1)
    params.put(mfmModelParam.k2, k2)
    params.put(mfmModelParam.initMean, initMean)
    params.put(mfmModelParam.initStdev, initStdev)
    params.put(mfmModelParam.maxInteractFeatures, maxInteractFeatures)
    params.put(mfmModelParam.numClasses, numClasses)
    // 返回MFM参数
    mfmModelParam
  }

  /**
    * 从本地文件载入参数
    *
    * @param location 本地文件位置
    * @param params 参数池
    * @return 分解机参数
    */
  def fromLocal(location: String, params: ParamMap): MfmModelParam = {
    // 初始化参数对象
    val mfmModelParam = new MfmModelParam {}
    implicit val formats = DefaultFormats
    // 读取参数值
    val paramValues = parse(File(location).contentAsString)
    val binaryThreshold = (paramValues \ mfmModelParam.binaryThreshold.name).extract[Double]
    val reg0 = (paramValues \ mfmModelParam.reg0.name).extract[Double]
    val reg1 = (paramValues \ mfmModelParam.reg1.name).extract[Double]
    val reg2 = (paramValues \ mfmModelParam.reg2.name).extract[Double]
    val numFeatures = (paramValues \ mfmModelParam.numFeatures.name).extract[Int]
    val numFactors = (paramValues \ mfmModelParam.numFactors.name).extract[Int]
    val k0 = (paramValues \ mfmModelParam.k0.name).extract[Boolean]
    val k1 = (paramValues \ mfmModelParam.k1.name).extract[Boolean]
    val k2 = (paramValues \ mfmModelParam.k2.name).extract[Boolean]
    val initMean = (paramValues \ mfmModelParam.initMean.name).extract[Double]
    val initStdev = (paramValues \ mfmModelParam.initStdev.name).extract[Double]
    val maxInteractFeatures = (paramValues \ mfmModelParam.maxInteractFeatures.name).extract[Int]
    val numClasses = (paramValues \ mfmModelParam.numClasses.name).extract[Int]
    // 设置参数值
    params.put(mfmModelParam.binaryThreshold, binaryThreshold)
    params.put(mfmModelParam.reg0, reg0)
    params.put(mfmModelParam.reg1, reg1)
    params.put(mfmModelParam.reg2, reg2)
    params.put(mfmModelParam.numFeatures, numFeatures)
    params.put(mfmModelParam.numFactors, numFactors)
    params.put(mfmModelParam.k0, k0)
    params.put(mfmModelParam.k1, k1)
    params.put(mfmModelParam.k2, k2)
    params.put(mfmModelParam.initMean, initMean)
    params.put(mfmModelParam.initStdev, initStdev)
    params.put(mfmModelParam.maxInteractFeatures, maxInteractFeatures)
    params.put(mfmModelParam.numClasses, numClasses)
    // 返回FM参数
    mfmModelParam
  }
}