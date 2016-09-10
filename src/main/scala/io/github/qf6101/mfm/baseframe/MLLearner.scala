package io.github.qf6101.mfm.baseframe

import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 16-9-9.
  */
abstract class MLLearner(val params: ParamMap) extends Serializable {
  /**
    * 更新参数池
    *
    * @param updatingParams 更新参数
    */
  def updateParams(updatingParams: ParamMap): Unit = {
    params ++= updatingParams
  }
}
