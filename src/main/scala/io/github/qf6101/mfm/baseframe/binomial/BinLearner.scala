package io.github.qf6101.mfm.baseframe.binomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.MLLearner
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD

/**
  * Created by qfeng on 15-3-27.
  */

/**
  * 模型学习器基类
  * @param params 参数池
  */
abstract class BinLearner(override val params: ParamMap) extends MLLearner(params) {
  /**
    * 训练对应模型
    * @param dataset 训练集
    * @return 模型
    */
  def train(dataset: RDD[(Double, SparseVector[Double])]): BinModel
}
