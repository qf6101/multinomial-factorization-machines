package io.github.qf6101.mfm.base

import breeze.linalg.SparseVector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.rdd.RDD

/**
  * Created by qfeng on 15-3-27.
  */

/**
  * 模型学习器基类
  * @param paramPool 参数池
  * @param imbalanceThreshold 不平衡数据阈值。
  *                           当负样本个数/正样本个数的比例超过阈值时，即视为不平衡，学习过程中按比例对梯度做惩罚。
  */
abstract class MLLearner(val paramPool: ParamMap,
                         val imbalanceThreshold: Double) extends Serializable {
  /**
    * 训练对应模型
    * @param dataset 训练集
    * @return 模型
    */
  def train(dataset: RDD[(Double, SparseVector[Double])]): MLModel

  /**
    * 更新参数池
    * @param updatingParams 更新参数
    */
  def updateParams(updatingParams: ParamMap): Unit = {
    paramPool ++= updatingParams
  }

  /**
    * 根据正负样本不平衡的情况（负样本数多于正样本数），设置负样本的惩罚比例
    * 通常用于惩罚梯度
    *
    * @param data 数据集
    * @return 负样本的惩罚比例
    */
  protected def calcNegativePenalty(data: RDD[(Double, SparseVector[Double])]): Double = {
    //如果阈值为0,则不做惩罚
    if (imbalanceThreshold <= 0.0) {
      return 1.0
    }

    val positiveData = data.filter(_._1 > 0)
    val negativeData = data.filter(_._1 <= 0)
    val positiveCount = positiveData.count().toDouble
    val negativeCount = negativeData.count().toDouble
    val ratio = positiveCount / negativeCount

    if (ratio <= imbalanceThreshold) {
      ratio
    } else {
      1.0
    }
  }
}
