package io.github.qf6101.mfm.logisticregression

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.binary.{BinLearner, BinModel}
import io.github.qf6101.mfm.baseframe.{MLLearner, MLModel}
import io.github.qf6101.mfm.optimization.{LBFGS, Updater}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
  * Created by qfeng on 15-4-7.
  */

/**
  * 逻辑斯蒂模型的LBFGS学习器
 *
  * @param params 参数池*
  * @param updater 参数更新器
  * @param initialCoeffs 初始参数
  */
class LrLearnLBFGS(override val params: ParamMap,
                   val updater: Updater,
                   val initialCoeffs: Option[VectorCoefficients] = None)
  extends BinLearner(params) with LrModelParam {
  val lg = new LogisticGradient(params)
  val lbfgs = new LBFGS(lg, updater, params)

  /**
    * 训练逻辑斯蒂模型
 *
    * @param dataSet 训练集
    * @return 逻辑斯蒂模型
    */
  override def train(dataSet: RDD[(Double, SparseVector[Double])]): BinModel = {
    dataSet.persist(StorageLevel.MEMORY_AND_DISK_SER)
    val inputCoeffs = initialCoeffs match {
      case Some(value) => value
      case None => new VectorCoefficients(dataSet.first()._2.length)
    }
    val coeffs = lbfgs.optimize(dataSet, inputCoeffs, params(reg))
    dataSet.unpersist()
    new LrModel(coeffs.asInstanceOf[VectorCoefficients], this, params)
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
    * @param params 参数池*
    * @param updater 参数更新器
    * @param initialCoeffs 初始参数
    * @return 逻辑斯蒂模型
    */
  def train(dataset: RDD[(Double, SparseVector[Double])],
            params: ParamMap,
            updater: Updater,
            initialCoeffs: Option[VectorCoefficients] = None): MLModel = {
    new LrLearnLBFGS(params, updater, initialCoeffs).train(dataset)
  }
}
