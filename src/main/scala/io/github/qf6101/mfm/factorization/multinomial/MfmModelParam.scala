package io.github.qf6101.mfm.factorization.multinomial

import io.github.qf6101.mfm.baseframe.ModelParam
import io.github.qf6101.mfm.baseframe.mutinomial.MultiModelParam
import io.github.qf6101.mfm.factorization.binomial.FmModelParam
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.sql.SparkSession
import org.json4s.JsonAST
import org.json4s.JsonDSL._


/**
  * Created by qfeng on 16-9-7.
  */
trait MfmModelParam extends FmModelParam with MultiModelParam {
  val numClasses: Param[Int] = new Param("MfmModelParam", "numClasses", "标签数目", ParamValidators.gt(0))

  /**
    * Transform parameters to json object
    *
    * @return parameters in json format
    */
  override def toJSON(params: ParamMap): JsonAST.JObject = {
    super.toJSON(params) ~
      (ModelParam.namingParamType -> this.getClass.toString()) ~
      (numClasses.name -> params(numClasses))
  }
}

object MfmModelParam {
  /**
    * 参数文件构造分解机模型参数
    *
    * @param location 文件位置
    * @param params 参数池
    * @return 分解机型参数
    */
  def apply(location: String, params: ParamMap): MfmModelParam = {
    val mfmModelParam = new MfmModelParam {}
    val spark = SparkSession.builder().getOrCreate()
    val paramValues = spark.read.json(location).first()
    val binaryThreshold = paramValues.getAs[Double](mfmModelParam.binaryThreshold.name)
    val reg0 = paramValues.getAs[Double](mfmModelParam.reg0.name)
    val reg1 = paramValues.getAs[Double](mfmModelParam.reg1.name)
    val reg2 = paramValues.getAs[Double](mfmModelParam.reg2.name)
    val numFeatures = paramValues.getAs[Int](mfmModelParam.numFeatures.name)
    val numFactors = paramValues.getAs[Int](mfmModelParam.numFactors.name)
    val k0 = paramValues.getAs[Boolean](mfmModelParam.k0.name)
    val k1 = paramValues.getAs[Boolean](mfmModelParam.k1.name)
    val k2 = paramValues.getAs[Boolean](mfmModelParam.k2.name)
    val initMean = paramValues.getAs[Double](mfmModelParam.initMean.name)
    val initStdev = paramValues.getAs[Double](mfmModelParam.initStdev.name)
    val maxInteractFeatures = paramValues.getAs[Int](mfmModelParam.maxInteractFeatures.name)
    val numClasses = paramValues.getAs[Int](mfmModelParam.numClasses.name)
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
    mfmModelParam
  }
}