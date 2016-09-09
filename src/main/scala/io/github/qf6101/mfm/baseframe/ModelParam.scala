package io.github.qf6101.mfm.baseframe

import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}

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
    * 将模型参数值转成字符串形式
    *
    * @param paramPool 参数池
    * @return 模型参数值的字符串形式
    */
  def mkString(paramPool: ParamMap): String
}
