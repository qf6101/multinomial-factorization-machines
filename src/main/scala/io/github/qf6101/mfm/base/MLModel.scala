package io.github.qf6101.mfm.base

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.tuning.BinaryClassificationMetrics
import io.github.qf6101.mfm.util.{HDFSUtil, Logging}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.SparkContext

import scala.util.control.Breaks

/**
  * Created by qfeng on 15-3-27.
  */

/**
  * 预测模型基类
  *
  * @param coeffs 模型系数
  * @param paramHolder 模型参赛
  * @param paramPool 参数池（保存参数的值）
  */
abstract class MLModel(val coeffs: Coefficients,
                       val paramHolder: ModelParam,
                       val paramPool: ParamMap) extends Logging with Serializable {
  //设置默认的阈值为0.5
  paramPool.put(paramHolder.binaryThreshold, 0.5)

  /**
    * 对输入数据进行预测
    *
    * @param data 输入数据
    * @return 预测值(0~1)
    */
  def regressionPredict(data: SparseVector[Double]): Double

  /**
    * 对输入数据集进行预测
    *
    * @param dataSet 输入数据集
    * @return 预测值集合(0~1)
    */
  def regressionPredict(dataSet: RDD[SparseVector[Double]]): RDD[Double] = {
    dataSet.map(regressionPredict)
  }

  /**
    * 对输入数据集进行预测
    *
    * @param dataSet 输入数据集
    * @return 预测值集合(0/1)
    */
  def classifPredict(dataSet: RDD[SparseVector[Double]]): RDD[Double] = {
    dataSet.map(classifPredict)
  }

  /**
    * 对输入数据进行预测
    *
    * @param data 输入数据
    * @return 预测值(0/1)
    */
  def classifPredict(data: SparseVector[Double]): Double = {
    val score = regressionPredict(data)
    if (score > paramPool(paramHolder.binaryThreshold)) {
      1.0
    } else {
      0.0
    }
  }

  /**
    * 保存模型
    *
    * @param attachedInfo 模型文件中附加的信息（例如learner和model等信息）
    * @param file 保存模型的位置
    */
  def saveModel(attachedInfo: String, file: String): Unit = {
    val sb = new StringBuilder(this.toString)
    //描述模型质量
    sb ++= "\n======== attached segment ========\n"
    sb ++= attachedInfo
    //保存模型
    HDFSUtil.deleteIfExists(file)
    SparkContext.getOrCreate().parallelize(Array(sb.toString())).repartition(1).saveAsTextFile(file)
  }

  /**
    * 将模型信息转成字符串，包含三部分信息：
    * （1）特征映射
    * （2）模型系数
    * （3）模型参数
    *
    * @return 字符串表示的模型信息
    */
  override def toString: String = {
    val sb = new StringBuilder()
    //描述模型系数
    val coeffString = coeffs.toString()
    sb ++= s"======= coefficient segment: ${coeffString.split("\n").length} ========\n"
    sb ++= coeffString
    sb += '\n'
    //描述模型参数
    sb ++= "======== parameter segment ========\n"
    sb ++= paramHolder.mkString(paramPool)
    //返回结果
    sb.toString()
  }

  /**
    * 选择二分分离器的阈值（固定AUC，选择F1-score最大的阈值）
    *
    * @param dataSet 数据集合
    */
  def selectThreshold(dataSet: RDD[(Double, SparseVector[Double])]): Array[BinaryClassificationMetrics] = {
    //生成对数据集的预测结果并持久化
    val scoreAndLabels = dataSet.map { case (label, data) =>
      (regressionPredict(data), label)
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
    paramPool.put(paramHolder.binaryThreshold, selectedThreshold)
    //计算最终的度量指标
    val finalMetrics = new BinaryClassificationMetrics(scoreAndLabels, selectedThreshold)
    logInfo(s"selected threshold: $selectedThreshold, metrics: ${finalMetrics.toString}}")
    //解除持久化
    scoreAndLabels.unpersist()
    Array(finalMetrics)
  }
}
