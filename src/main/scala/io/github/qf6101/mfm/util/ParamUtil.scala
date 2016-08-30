package io.github.qf6101.mfm.util

import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 15-3-31.
  */

/**
  * 参数工具类实例
  */
object ParamUtil {

  /**
    * 参数池转成字符串
    * @param paramPool 参数池
    * @return 字符串
    */
  def paramPoolToString(paramPool: ParamMap): String = {
    paramPool.toSeq.map { paramPair => paramPair.value match {
      case v: Array[_] => s"${paramPair.param.name}:${v.mkString(",")}"
      case _ => s"${paramPair.param.name}:${paramPair.value}"
    }
    }.mkString(", ")
  }
}
