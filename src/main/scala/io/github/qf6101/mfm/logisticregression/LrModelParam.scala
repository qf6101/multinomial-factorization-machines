package io.github.qf6101.mfm.logisticregression

import io.github.qf6101.mfm.baseframe.ModelParam
import io.github.qf6101.mfm.baseframe.binary.BinModelParam
import org.apache.spark.ml.param.{Param, ParamMap}


/**
  * Created by qfeng on 15-3-18.
  */

/**
  * 逻辑斯蒂模型的参数
  */
trait LrModelParam extends BinModelParam {
  val reg: Param[Array[Double]] = new Param("LrModelParam", "reg", "正则参数")

  /**
    * 将模型参数值转成字符串形式
    *
    * @param paramPool 参数池
    * @return 模型参数值的字符串形式
    */
  override def mkString(paramPool: ParamMap): String = {
    val sb = new StringBuilder()
    sb ++= "binaryThreshold:"
    sb ++= "%1.2f".format(paramPool(binaryThreshold))
    sb ++= " reg:"
    sb ++= paramPool(reg).mkString(", ")
    sb ++= " initMean:"
    sb ++= paramPool(initMean).toString
    sb ++= " initStdev:"
    sb ++= paramPool(initStdev).toString
    sb.toString()
  }
}

object LrModelParam {
  /**
    * 根据字符串数组构造逻辑斯蒂模型参数
    *
    * @param content 字符串
    * @param paramPool 参数池
    * @return 逻辑斯蒂模型参数
    */
  def apply(content: String, paramPool: ParamMap): LrModelParam = {
    val lrModelParam = new LrModelParam {}
    val codeArray = content.split(" ")
    val binaryThreshold = codeArray(0).split(":")(1).trim.toDouble
    val reg = codeArray(1).split(":")(1).split(",").map(_.trim.toDouble)
    val initMean = codeArray(2).split(":")(1).trim.toDouble
    val initStdev = codeArray(3).split(":")(1).trim.toDouble
    paramPool.put(lrModelParam.binaryThreshold, binaryThreshold)
    paramPool.put(lrModelParam.reg, reg)
    paramPool.put(lrModelParam.initMean, initMean)
    paramPool.put(lrModelParam.initStdev, initStdev)
    lrModelParam
  }
}
