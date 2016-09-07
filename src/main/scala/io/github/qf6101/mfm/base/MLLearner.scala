package io.github.qf6101.mfm.base

import breeze.linalg.SparseVector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD

/**
  * Created by qfeng on 15-3-27.
  */

/**
  * 模型学习器基类
  * @param paramPool 参数池
  */
abstract class MLLearner(val paramPool: ParamMap) extends Serializable {
  /**
    * 训练对应模型
    * @param dataset 训练集
    * @return 模型
    */
  def train(dataset: RDD[(Double, SparseVector[Double])]): MLModel

  /**
    * 更新参数池
    * @param updatingParams 更新参数
    */
  def updateParams(updatingParams: ParamMap): Unit = {
    paramPool ++= updatingParams
  }
}
