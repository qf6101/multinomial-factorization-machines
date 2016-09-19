package io.github.qf6101.mfm.logisticregression

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.MLModel
import io.github.qf6101.mfm.baseframe.binomial.{BinLearner, BinModel}
import io.github.qf6101.mfm.optimization.{GradientDescent, Updater}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
  * Created by qfeng on 15-3-17.
  */

/**
  * Train a classification model for Logistic Regression using Stochastic Gradient Descent. By
  * default L2 regularization is used, which can be changed via
  * [[LrLearnSGD]].
  */

/**
  * 逻辑斯蒂模型的SGD学习器
  *
  * @param params        参数池*
  * @param updater       参数更新器
  * @param initialCoeffs 初始参数
  */
class LrLearnSGD(override val params: ParamMap,
                 val updater: Updater,
                 val initialCoeffs: Option[VectorCoefficients] = None)
  extends BinLearner(params) with LrModelParam {
  val lg = new LogisticGradient(params)
  val gd = new GradientDescent(lg, updater, params)

  /**
    * 训练逻辑斯蒂模型
    *
    * @param dataSet 训练集
    * @return 逻辑斯蒂模型
    */
  override def train(dataSet: RDD[(Double, SparseVector[Double])]): BinModel = {
    dataSet.persist(StorageLevel.MEMORY_AND_DISK_SER_2)
    val inputCoeffs = initialCoeffs match {
      case Some(value) => value
      case None => new VectorCoefficients(dataSet.first()._2.length)
    }
    val coeffs = gd.optimize(dataSet, inputCoeffs, params(reg))
    dataSet.unpersist()
    new LrModel(this, coeffs.asInstanceOf[VectorCoefficients], params)
  }


}

/**
  * 逻辑斯蒂模型的SGD学习器实例
  */
object LrLearnSGD {
  /**
    * 训练逻辑斯蒂模型
    *
    * @param dataset       数据集
    * @param params        参数池*
    * @param updater       参数更新器
    * @param initialCoeffs 初始参数
    * @return 逻辑斯蒂模型
    */
  def train(dataset: RDD[(Double, SparseVector[Double])],
            params: ParamMap,
            updater: Updater,
            initialCoeffs: Option[VectorCoefficients] = None): MLModel = {
    new LrLearnSGD(params, updater, initialCoeffs).train(dataset)
  }
}