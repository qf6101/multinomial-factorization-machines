package io.github.qf6101.mfm.optimization

import breeze.linalg.SparseVector
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}
import io.github.qf6101.mfm.base.Coefficients
import io.github.qf6101.mfm.regression.VectorCoefficients
import io.github.qf6101.mfm.util.Logging
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD

import scala.collection.mutable.{ArrayBuffer, HashMap}

/**
  * Created by qfeng on 15-4-7.
  */


class LBFGS(private var gradient: Gradient, private var updater: Updater, private var paramPool: ParamMap) extends
Optimizer with LBFGSParam with Logging {
  override def optimize(data: RDD[(Double, SparseVector[Double])],
                        initialCoeffs: Coefficients,
                        reg: Array[Double],
                        negativePenalty: Double):
  Coefficients = {
    val (coeffs, _) = optimizeWithHistory(data, initialCoeffs, reg, negativePenalty)
    coeffs
  }

  def optimizeWithHistory(data: RDD[(Double, SparseVector[Double])],
                          initialCoeffs: Coefficients,
                          reg: Array[Double],
                          negativePenalty: Double = 1.0):
  (Coefficients, Array[Double]) = {
    //获取参数
    val numIterationsValue = paramPool(numIterations)
    val numCorrectionsValue = paramPool(numCorrections)
    val convergenceTolValue = paramPool(convergenceTol)
    //初始化损失值数组、数据集大小
    val lossHistory = new ArrayBuffer[Double](numIterationsValue)
    val numExamples = data.count()
    //初始化系数、损失函数形式
    val vecInitialCoeffs = initialCoeffs.asInstanceOf[VectorCoefficients]
    val costFun = new CostFun(data, gradient, updater, reg, numExamples, negativePenalty)
    val lbfgs = new BreezeLBFGS[SparseVector[Double]](numIterationsValue, numCorrectionsValue, convergenceTolValue)
    //创建LBFGS状态序列
    val states = lbfgs.iterations(new CachedDiffFunction(costFun), VCToBSV(vecInitialCoeffs))
    //执行迭代
    var i = 0
    var state = states.next()
    while (states.hasNext) {
      i += 1
      logDebug(s"Iteration ($i/${numIterationsValue}) loss: ${state.value}")
      lossHistory.append(state.value)
      state = states.next()
    }
    lossHistory.append(state.value)
    //返回结果
    (state.x, lossHistory.toArray)
  }

  /**
    * 向量系数转成breeze的稀疏向量
    *
    * @param in 向量系数
    * @return breeze的稀疏向量
    */
  implicit def VCToBSV(in: VectorCoefficients): SparseVector[Double] = {
    val out = SparseVector.zeros[Double](in.size + 1)
    out.update(0, in.w0)
    in.w.foreach { case (index, value) =>
      out.update(index + 1, value)
    }
    out
  }

  /**
    * breeze的稀疏向量转成向量系数
    *
    * @param in breeze的稀疏向量
    * @return 向量系数
    */
  implicit def BSVToVC(in: SparseVector[Double]): VectorCoefficients = {
    val w0 = in(0)
    val w = HashMap[Int, Double]()
    in.activeIterator.foreach { case (index, value) =>
      if (index != 0) {
        w += (index - 1) -> value
      }
    }
    new VectorCoefficients(in.length - 1, w0, w)
  }

  /**
    * CostFun implements Breeze's DiffFunction[T], which returns the loss and gradient
    * at a particular point (weights). It's used in Breeze's convex optimization routines.
    */
  private class CostFun(data: RDD[(Double, SparseVector[Double])],
                        gradient: Gradient,
                        updater: Updater,
                        reg: Array[Double],
                        numExamples: Long,
                        negativePenalty: Double) extends DiffFunction[SparseVector[Double]] with Serializable {

    override def calculate(weights: SparseVector[Double]): (Double, SparseVector[Double]) = {
      // Have a local copy to avoid the serialization of CostFun object which is not serializable.
      val w = weights.copy
      val n = weights.length
      val bcW = data.context.broadcast(w)
      val localGradient = gradient

      val (gradientSum, lossSum) = data.treeAggregate((new VectorCoefficients(n - 1), 0.0))(
        seqOp = (c, v) => (c, v) match {
          case ((grad, loss), (label, features)) =>
            val l = localGradient.compute(features, label, bcW.value, grad, negativePenalty)
            (grad, loss + l)
        },
        combOp = (c1, c2) => (c1, c2) match {
          case ((grad1, loss1), (grad2, loss2)) =>
            grad1 += grad2
            (grad1, loss1 + loss2)
        })

      /**
        * regVal is sum of weight squares if it's L2 updater;
        * for other updater, the same logic is followed.
        */
      val regVal = updater.compute(w, new VectorCoefficients(n - 1), 0, 1, reg)._2
      val loss = lossSum / numExamples + regVal
      /**
        * It will return the gradient part of regularization using updater.
        *
        * Given the input parameters, the updater basically does the following,
        *
        * w' = w - thisIterStepSize * (gradient + regGradient(w))
        * Note that regGradient is function of w
        *
        * If we set gradient = 0, thisIterStepSize = 1, then
        *
        * regGradient(w) = w - w'
        *
        * TODO: We need to clean it up by separating the logic of regularization out
        * from updater to regularizer.
        */
      // The following gradientTotal is actually the regularization part of gradient.
      // Will add the gradientSum computed from the data with weights in the next step.
      val gradientTotal = BSVToVC(w)
      gradientTotal -= updater.compute(w, new VectorCoefficients(n - 1), 1, 1, reg)._1.asInstanceOf[VectorCoefficients]
      gradientTotal += gradientSum * (1.0 / numExamples)

      (loss, gradientTotal)
    }
  }
}
