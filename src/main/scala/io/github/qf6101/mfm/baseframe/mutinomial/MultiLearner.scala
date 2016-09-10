package io.github.qf6101.mfm.baseframe.mutinomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.MLLearner
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD

/**
  * Created by qfeng on 16-9-9.
  */
abstract class MultiLearner(override val params: ParamMap) extends MLLearner(params) {
  /**
    * 训练对应模型
    * @param dataset 训练集
    * @return 模型
    */
  def train(dataset: RDD[(Double, SparseVector[Double])]): MultiModel
}
