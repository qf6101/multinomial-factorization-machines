package io.github.qf6101.mfm.factorization

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.base.{MLLearner, MLModel}
import io.github.qf6101.mfm.optimization.{GradientDescent, Updater}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD

/**
  * Created by qfeng on 15-3-27.
  */

/**
  * FM模型的SGD学习器
  * @param paramPool 参数池
  * @param updater 模型参数更新器
  */
class FmLearnSGD(override val paramPool: ParamMap,
                 val updater: Updater)
  extends MLLearner(paramPool) with FmModelParam {
  val lg = new FmGradient(this, paramPool)
  val gd = new GradientDescent(lg, updater, paramPool)

  /**
    * 训练模型
    * @param dataset 训练集
    * @return 模型
    */
  override def train(dataset: RDD[(Double, SparseVector[Double])]): MLModel = {
    val initialCoeffs = new FmCoefficients(paramPool(initMean), paramPool(initStdev),
      paramPool(numFeatures), paramPool(maxInteractFeatures), paramPool(numFactors), paramPool(k0), paramPool(k1), paramPool(k2))
    val regs = Array(paramPool(reg0), paramPool(reg1), paramPool(reg2))
    val coeffs = gd.optimize(dataset, initialCoeffs, regs)
    new FmModel(coeffs.asInstanceOf[FmCoefficients], this, paramPool)
  }
}

/**
  * FM模型的SGD学习器实例
  */
object FmLearnSGD {
  def train(dataset: RDD[(Double, SparseVector[Double])],
            paramPool: ParamMap,
            updater: Updater): MLModel = {
    new FmLearnSGD(paramPool, updater).train(dataset)
  }
}
