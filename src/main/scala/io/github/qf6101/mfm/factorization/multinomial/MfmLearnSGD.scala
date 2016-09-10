package io.github.qf6101.mfm.factorization.multinomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.mutinomial.{MultiLearner, MultiModel}
import io.github.qf6101.mfm.factorization.binary.{FmCoefficients, FmGradient, FmModel}
import io.github.qf6101.mfm.optimization.{GradientDescent, Updater}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD

/**
  * Created by qfeng on 16-9-7.
  */
class MfmLearnSGD(override val params: ParamMap,
                  val updater: Updater) extends MultiLearner(params) with MfmModelParam {
  val lg = new MfmGradient(this, params)
  val gd = new GradientDescent(lg, updater, params)
  /**
    * 训练对应模型
    *
    * @param dataset 训练集
    * @return 模型
    */
  override def train(dataset: RDD[(Double, SparseVector[Double])]): MultiModel = {
    val initialCoeffs = new MfmCoefficients(params(initMean), params(initStdev), params(numFeatures),
      params(maxInteractFeatures), params(numFactors), params(k0), params(k1), params(k2), params(numClasses))
    val regs = Array(params(reg0), params(reg1), params(reg2))
    val coeffs = gd.optimize(dataset, initialCoeffs, regs)
    new MfmModel(coeffs.asInstanceOf[MfmCoefficients], this, params)
  }
}
