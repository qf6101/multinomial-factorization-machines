package io.github.qf6101.mfm.baseframe.binomial

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.{Coefficients, MLModel}
import io.github.qf6101.mfm.tuning.BinaryClassificationMetrics
import io.github.qf6101.mfm.util.Logging
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.util.control.Breaks

/**
  * Created by qfeng on 15-3-27.
  */

/**
  * 预测模型基类
  *
  * @param paramMeta 模型参赛
  * @param coeffs    模型系数
  * @param params    参数池（保存参数的值）
  */
abstract class BinModel(override val paramMeta: BinModelParam,
                        override val coeffs: Coefficients,
                        override val params: ParamMap)
  extends MLModel(paramMeta, coeffs, params) with Logging with Serializable {
  //设置默认的阈值为0.5
  params.put(paramMeta.binaryThreshold, 0.5)

  /**
    * 对输入数据进行预测
    *
    * @param data 输入数据
    * @return 预测值(0~1)
    */
  def predict(data: SparseVector[Double]): Double

  /**
    * 对输入数据集进行预测
    *
    * @param dataSet 输入数据集
    * @return 预测值集合(0~1)
    */
  def predict(dataSet: RDD[SparseVector[Double]]): RDD[Double] = {
    dataSet.map(predict)
  }

  /**
    * 选择二分分离器的阈值（固定AUC，选择F1-score最大的阈值）
    *
    * @param dataSet 数据集合
    */
  def selectThreshold(dataSet: RDD[(Double, SparseVector[Double])]): Array[BinaryClassificationMetrics] = {
    //生成对数据集的预测结果并持久化
    val scoreAndLabels = dataSet.map { case (label, data) =>
      (predict(data), label)
    }.persist(StorageLevel.MEMORY_AND_DISK_SER)
    //以0.05为间隔，尝试每个threshold，选择F1_score最大的threshold
    //直至遇到F1_score为NaN，停止尝试
    var maxF1Score = Double.MinValue
    var selectedThreshold = 0.5
    val loop = new Breaks
    loop.breakable {
      for (tryThreshold <- 0.05 until 1.0 by 0.05) {
        val metrics = new BinaryClassificationMetrics(scoreAndLabels, tryThreshold)
        logDebug(s"threshold selection => f1-score: ${"%1.4f".format(metrics.f1_scores._1)}, threshold: ${"%1.2f".format(tryThreshold)}")
        if (metrics.f1_scores._1.isNaN) {
          loop.break()
        } else if (metrics.f1_scores._1 > maxF1Score) {
          maxF1Score = metrics.f1_scores._1
          selectedThreshold = tryThreshold
        }
      }
    }
    //设置选择得到的threshold
    params.put(paramMeta.binaryThreshold, selectedThreshold)
    //计算最终的度量指标
    val finalMetrics = new BinaryClassificationMetrics(scoreAndLabels, selectedThreshold)
    logInfo(s"selected threshold: $selectedThreshold, metrics: ${finalMetrics.toString}}")
    //解除持久化
    scoreAndLabels.unpersist()
    Array(finalMetrics)
  }
}
