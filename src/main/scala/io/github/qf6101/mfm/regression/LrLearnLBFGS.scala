package io.github.qf6101.mfm.regression

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.base.{MLLearner, MLModel}
import io.github.qf6101.mfm.optimization.{LBFGS, Updater}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
  * Created by qfeng on 15-4-7.
  */

/**
  * 逻辑斯蒂模型的LBFGS学习器
  * @param paramPool 参数池*
  * @param updater 参数更新器
  * @param initialCoeffs 初始参数
  */
class LrLearnLBFGS(override val paramPool: ParamMap,
                   val updater: Updater,
                   val initialCoeffs: Option[VectorCoefficients] = None)
  extends MLLearner(paramPool) with LrModelParam {
  val lg = new LogisticGradient(paramPool)
  val lbfgs = new LBFGS(lg, updater, paramPool)

  /**
    * 训练逻辑斯蒂模型
    * @param dataSet 训练集
    * @return 逻辑斯蒂模型
    */
  override def train(dataSet: RDD[(Double, SparseVector[Double])]): MLModel = {
    dataSet.persist(StorageLevel.MEMORY_AND_DISK_SER)
    val inputCoeffs = initialCoeffs match {
      case Some(value) => value
      case None => new VectorCoefficients(dataSet.first()._2.length)
    }
    val coeffs = lbfgs.optimize(dataSet, inputCoeffs, paramPool(reg))
    dataSet.unpersist()
    new LrModel(coeffs.asInstanceOf[VectorCoefficients], this, paramPool)
  }
}

/**
  * 逻辑斯蒂模型的LBFGS学习器实例
  */
object LrLearnLBFGS {

  /**
    * 训练逻辑斯蒂模型
    *
    * @param dataset 数据集
    * @param paramPool 参数池*
    * @param updater 参数更新器
    * @param initialCoeffs 初始参数
    * @return 逻辑斯蒂模型
    */
  def train(dataset: RDD[(Double, SparseVector[Double])],
            paramPool: ParamMap,
            updater: Updater,
            initialCoeffs: Option[VectorCoefficients] = None): MLModel = {
    new LrLearnLBFGS(paramPool, updater, initialCoeffs).train(dataset)
  }
}
