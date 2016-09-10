package io.github.qf6101.mfm.factorization.binomial

import io.github.qf6101.mfm.baseframe.binomial.BinModelParam
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}

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
    * 将模型参数值转成字符串形式
    *
    * @param params 参数池
    * @return 模型参数值的字符串形式
    */
  override def mkString(params: ParamMap): String = {
    val sb = new StringBuilder()
    sb ++= "binaryThreshold:"
    sb ++= "%1.2f".format(params(binaryThreshold))
    sb ++= " reg0:"
    sb ++= params(reg0).toString
    sb ++= " reg1:"
    sb ++= params(reg1).toString
    sb ++= " reg2:"
    sb ++= params(reg2).toString
    sb ++= " numFeatures:"
    sb ++= params(numFeatures).toString
    sb ++= " numFactors:"
    sb ++= params(numFactors).toString
    sb ++= " k0:"
    sb ++= params(k0).toString
    sb ++= " k1:"
    sb ++= params(k1).toString
    sb ++= " k2:"
    sb ++= params(k2).toString
    sb ++= " initMean:"
    sb ++= params(initMean).toString
    sb ++= " initStdev:"
    sb ++= params(initStdev).toString
    sb ++= " maxInteractFeatures:"
    sb ++= params(maxInteractFeatures).toString
    sb.toString()
  }
}

object FmModelParam {
  /**
    * 根据字符串数组构造分解机模型参数
    *
    * @param content 字符串
    * @param params 参数池
    * @return 分解机型参数
    */
  def apply(content: String, params: ParamMap): FmModelParam = {
    val fmModelParam = new FmModelParam {}
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
    params.put(fmModelParam.binaryThreshold, binaryThreshold)
    params.put(fmModelParam.reg0, reg0)
    params.put(fmModelParam.reg1, reg1)
    params.put(fmModelParam.reg2, reg2)
    params.put(fmModelParam.numFeatures, numAttrs)
    params.put(fmModelParam.numFactors, numFactors)
    params.put(fmModelParam.k0, k0)
    params.put(fmModelParam.k1, k1)
    params.put(fmModelParam.k2, k2)
    params.put(fmModelParam.initMean, initMean)
    params.put(fmModelParam.initStdev, initStdev)
    params.put(fmModelParam.maxInteractFeatures, maxInteractAttr)
    fmModelParam
  }
}