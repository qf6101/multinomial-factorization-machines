package io.github.qf6101.mfm.factorization.binomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.MLModel
import io.github.qf6101.mfm.baseframe.binomial.{BinLearner, BinModel}
import io.github.qf6101.mfm.optimization.{GradientDescent, Updater}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD

/**
  * Created by qfeng on 15-3-27.
  */

/**
  * FM模型的SGD学习器
 *
  * @param params 参数池
  * @param updater 模型参数更新器
  */
class FmLearnSGD(override val params: ParamMap,
                 val updater: Updater)
  extends BinLearner(params) with FmModelParam {
  val lg = new FmGradient(this, params)
  val gd = new GradientDescent(lg, updater, params)

  /**
    * 训练模型
 *
    * @param dataset 训练集
    * @return 模型
    */
  override def train(dataset: RDD[(Double, SparseVector[Double])]): BinModel = {
    val initialCoeffs = new FmCoefficients(params(initMean), params(initStdev),
      params(numFeatures), params(maxInteractFeatures), params(numFactors), params(k0), params(k1), params(k2))
    val regs = Array(params(reg0), params(reg1), params(reg2))
    val coeffs = gd.optimize(dataset, initialCoeffs, regs)
    new FmModel(this, coeffs.asInstanceOf[FmCoefficients], params)
  }
}

/**
  * FM模型的SGD学习器实例
  */
object FmLearnSGD {
  def train(dataset: RDD[(Double, SparseVector[Double])],
            params: ParamMap,
            updater: Updater): MLModel = {
    new FmLearnSGD(params, updater).train(dataset)
  }
}
