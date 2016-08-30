package io.github.qf6101.mfm.tuning

import org.apache.spark.mllib.evaluation.{RegressionMetrics => RM}
import org.apache.spark.rdd.RDD

/**
  * User: qfeng
  * Date: 15-8-11 下午4:03
  * Usage:
  */
class RegressionMetrics(val scoreAndLabels: RDD[(Double, Double)]) {
  private val rm = new RM(scoreAndLabels)

  /**
    * 将各个度量指标转成字符串形式
    *
    * @return MSE
    */
  override def toString: String = {
    val result = new StringBuilder
    result.append("MSE: ")
    result.append(MSE)
    result.toString()
  }

  def MSE: Double = {
    rm.meanSquaredError
  }
}
