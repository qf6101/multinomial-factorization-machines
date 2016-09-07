package io.github.qf6101.mfm.factorization

import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}

/**
  * Created by qfeng on 16-9-7.
  */
class MfmModelParam extends FmModelParam {
  val numClasses: Param[Int] = new Param("MfmModelParam", "numClasses", "标签数目", ParamValidators.gt(0))

  /**
    * 将模型参数值转成字符串形式
    *
    * @param paramPool 参数池
    * @return 模型参数值的字符串形式
    */
  override def mkString(paramPool: ParamMap): String = {
    val sb = new StringBuilder(super.mkString(paramPool))
    sb ++= " numClasses:"
    sb ++= paramPool(numClasses).toString
    sb.toString()
  }
}

object MfmModelParam {
  /**
    * 根据字符串数组构造分解机模型参数
    *
    * @param content 字符串
    * @param paramPool 参数池
    * @return 分解机型参数
    */
  def apply(content: String, paramPool: ParamMap): MfmModelParam = {
    val mfmModelParam = new MfmModelParam {}
    val codeArray = content.split(" ")
    val binaryThreshold = codeArray(0).split(":")(1).trim.toDouble
    val reg0 = codeArray(1).split(":")(1).trim.toDouble
    val reg1 = codeArray(2).split(":")(1).trim.toDouble
    val reg2 = codeArray(3).split(":")(1).trim.toDouble
    val numAttrs = codeArray(4).split(":")(1).trim.toInt
    val numFactors = codeArray(5).split(":")(1).trim.toInt
    val k0 = codeArray(6).split(":")(1).trim.toBoolean
    val k1 = codeArray(7).split(":")(1).trim.toBoolean
    val k2 = codeArray(8).split(":")(1).trim.toBoolean
    val initMean = codeArray(9).split(":")(1).trim.toDouble
    val initStdev = codeArray(10).split(":")(1).trim.toDouble
    val maxInteractAttr = codeArray(11).split(":")(1).trim.toInt
    val numClasses = codeArray(12).split(":")(1).trim.toInt
    paramPool.put(mfmModelParam.binaryThreshold, binaryThreshold)
    paramPool.put(mfmModelParam.reg0, reg0)
    paramPool.put(mfmModelParam.reg1, reg1)
    paramPool.put(mfmModelParam.reg2, reg2)
    paramPool.put(mfmModelParam.numFeatures, numAttrs)
    paramPool.put(mfmModelParam.numFactors, numFactors)
    paramPool.put(mfmModelParam.k0, k0)
    paramPool.put(mfmModelParam.k1, k1)
    paramPool.put(mfmModelParam.k2, k2)
    paramPool.put(mfmModelParam.initMean, initMean)
    paramPool.put(mfmModelParam.initStdev, initStdev)
    paramPool.put(mfmModelParam.maxInteractFeatures, maxInteractAttr)
    paramPool.put(mfmModelParam.numClasses, numClasses)
    mfmModelParam
  }
}