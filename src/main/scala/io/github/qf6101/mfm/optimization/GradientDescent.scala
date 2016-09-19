package io.github.qf6101.mfm.optimization

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.Coefficients
import io.github.qf6101.mfm.util.Logging
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * Created by qfeng on 15-3-11.
  */

/**
  * 随机梯度下降器
  *
  * @param gradient 梯度逻辑
  * @param updater  更新逻辑
  * @param params   参数池
  */
class GradientDescent(private var gradient: Gradient, private var updater: Updater, private var params: ParamMap)
  extends Optimizer with SGDParam with Logging {

  /**
    * 最优化函数
    *
    * @param data          样本数据集
    * @param initialCoeffs 初始化系数值
    * @param regParam      正则参数值
    * @return 学习后的参数
    */
  override def optimize(data: RDD[(Double, SparseVector[Double])],
                        initialCoeffs: Coefficients,
                        regParam: Array[Double]): Coefficients = {
    val (coeffs, _) = optimizeWithHistory(data, initialCoeffs, regParam)
    coeffs
  }

  /**
    * 最优化函数
    *
    * @param data          样本数据集
    * @param initialCoeffs 初始化系数值
    * @param regParam      正则参数值
    * @return 学习后的参数
    */
  def optimizeWithHistory(data: RDD[(Double, SparseVector[Double])],
                          initialCoeffs: Coefficients,
                          regParam: Array[Double]): (Coefficients, Array[Double]) = {
    //获取参数
    val numIterationsValue = params(numIterations)
    val miniBatchFractionValue = params(miniBatchFraction)
    val stepSizeValue = params(stepSize)
    //初始化系数、正则值
    var coeffs = initialCoeffs.copy
    var regVal = updater.compute(coeffs, coeffs.copyEmpty(), 0, 1, regParam)._2
    val lossHistory = new ArrayBuffer[Double](numIterationsValue)
    //初始化临时变量：迭代次数、是否收敛、上次损失值
    var i = 0
    var reachStopCondition = false
    //开始迭代训练
    while (!reachStopCondition && i < numIterationsValue) {
      i += 1
      val bcCoeffs = SparkContext.getOrCreate.broadcast(coeffs)
      val (gradientSum, lossSum, miniBatchSize) = data.sample(withReplacement = false, miniBatchFractionValue, 42 + i)
        .treeAggregate(initialCoeffs.copyEmpty(), 0.0, 0L)(
          seqOp = (c, v) => {
            // c: (grad, loss, count), v: (label, features)
            val l = gradient.compute(v._2, v._1, bcCoeffs.value, c._1)
            (c._1, c._2 + l, c._3 + 1)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss, count)
            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })

      if (miniBatchSize > 0) {
        //计算损失值、新的系数、正则值
        lossHistory.append(lossSum / miniBatchSize + regVal)
        val update = updater.compute(coeffs, gradientSum / miniBatchSize.toDouble, stepSizeValue, i, regParam)
        //判断是否达到收敛条件
        val (converged, solutionDiff) = isConverged(update._1, coeffs)
        reachStopCondition = converged
        //更新系数和正则值
        coeffs = update._1
        regVal = update._2
        //打印调试信息：损失值
        logInfo(s"Iteration ($i/$numIterationsValue) loss: ${lossSum / miniBatchSize} and $regVal, solutionDiff: $solutionDiff")
      } else {
        logWarning(s"Iteration ($i/$numIterationsValue}). The size of sampled batch is zero")
      }
    }
    (coeffs, lossHistory.toArray)
  }

  /**
    * 判断是否达到收敛条件
    *
    * @param newCoeffs 更新后的系数
    * @param oldCoeffs 更新前的系数
    * @return 是否达到收敛条件
    */
  private def isConverged(newCoeffs: Coefficients, oldCoeffs: Coefficients): (Boolean, Double) = {
    val solutionDiff = newCoeffs.normDiff(oldCoeffs)
    (solutionDiff < params(convergenceTol) * Math.max(newCoeffs.norm, 1.0), solutionDiff)
  }
}
