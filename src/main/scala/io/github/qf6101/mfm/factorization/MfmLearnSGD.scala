package io.github.qf6101.mfm.factorization

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.base.{MLLearner, MLModel}
import io.github.qf6101.mfm.optimization.Updater
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD

/**
  * Created by qfeng on 16-9-7.
  */
class MfmLearnSGD(override val paramPool: ParamMap,
                  val updater: Updater) extends MLLearner(paramPool) with FmModelParam {
  /**
    * 训练对应模型
    *
    * @param dataset 训练集
    * @return 模型
    */
  override def train(dataset: RDD[(Double, SparseVector[Double])]): MLModel = ???
}
