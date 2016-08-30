package io.github.qf6101.mfm.tuning

import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics => BCM}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
  * User: qfeng
  * Date: 15-8-10 上午11:10
  * Usage: Binary classification evaluation, See https://en.wikipedia.org/wiki/Receiver_operating_characteristic
  */

class BinaryClassificationMetrics(private val rawScoreAndLabels: RDD[(Double, Double)],
                                  val threshold: Double = 0.5) extends Serializable {
  private val scoreAndLabels = rawScoreAndLabels.map { case(score, label) =>
    if(label <= 0) (score, 0.0) else (score, 1.0)
  }.persist(StorageLevel.MEMORY_AND_DISK_SER)

  private val metrics = computeMetrics
  val accuracy = metrics._1
  val precisions = (metrics._2, metrics._3)
  val recalls = (metrics._4, metrics._5)
  val f1_scores = (metrics._6, metrics._7)
  val AUC = AUCValue
  private lazy val AUCValue = computeAUC(metrics._8)

  /**
    * 将各个度量指标转成字符串形式（保留4位小数）
    *
    * @return (AUC, accuracy, precisions, recalls, f1_scores)
    */
  override def toString: String = {
    val result = new StringBuilder
    result.append("AUC: ")
    result.append("%1.4f".format(AUC))
    result.append(", accuracy: ")
    result.append("%1.4f".format(accuracy))
    result.append(", precisions: ")
    result.append(mkTupleString(precisions))
    result.append(", recalls: ")
    result.append(mkTupleString(recalls))
    result.append(", f1_scores: ")
    result.append(mkTupleString(f1_scores))
    result.toString()
  }

  /**
    * double类型的元组转成字符串（转成4位小数）
    *
    * @param t 元组
    * @return 4位小数表示的字符串
    */
  private def mkTupleString(t: (Double, Double)): String = {
    val result = new StringBuilder
    result.append("(")
    result.append("%1.4f".format(t._1))
    result.append(", ")
    result.append("%1.4f".format(t._2))
    result.append(")")
    result.toString()
  }

  private def computeAUC(numData: Int): Double = {
    var auc: Double = 0.0
    if (numData > 300000) {
      auc = new BCM(scoreAndLabels, 100000).areaUnderROC()
    } else {
      auc = new BCM(scoreAndLabels).areaUnderROC()
    }
    if(scoreAndLabels.getStorageLevel == StorageLevel.MEMORY_AND_DISK_SER) {
      scoreAndLabels.unpersist()
    }
    auc
  }

  /**
    * 计算各种衡量二分类模型的度量指标
    *
    * @return 指标依次为：(accuracy, positive precision, negative precision, positive recall, negative recall, positive f1_scores, negative f1_score)
    */
  private def computeMetrics: (Double, Double, Double, Double, Double, Double, Double, Int) = {
    val sc = scoreAndLabels.context
    val totalAccum = sc.accumulator(0)
    val testPositiveAccum = sc.accumulator(0)
    val condPositiveAccum = sc.accumulator(0)
    val truePositiveAccum = sc.accumulator(0)
    val trueNegativeAccum = sc.accumulator(0)

    scoreAndLabels.foreach { case (score, label) =>
      totalAccum += 1
      if (score > threshold) {
        testPositiveAccum += 1
      }
      if (label == 1.0) {
        condPositiveAccum += 1
      }
      if (score >= threshold && label == 1.0) {
        truePositiveAccum += 1
      }
      if (score < threshold && label == 0.0) {
        trueNegativeAccum += 1
      }
    }

    val totalNum = totalAccum.value.toDouble
    val testPositiveNum = testPositiveAccum.value.toDouble
    val testNegativeNum = totalNum - testPositiveNum
    val condPositiveNum = condPositiveAccum.value.toDouble
    val condNegativeNum = totalNum - condPositiveNum
    val truePositiveNum = truePositiveAccum.value.toDouble
    val trueNegativeNum = trueNegativeAccum.value.toDouble

    //accuracy
    val ACC = (truePositiveNum + trueNegativeNum) / totalNum
    //positive predictive value (positive precision)
    val PPV = truePositiveNum / testPositiveNum
    //negative predictive value (negative precision)
    val NPV = trueNegativeNum / testNegativeNum
    //true positive rate (sensitivity, positive recall)
    val TPR = truePositiveNum / condPositiveNum
    //true negative rate (specificity, negative recall)
    val TNR = trueNegativeNum / condNegativeNum
    //positive f1 score
    val F1P = (2 * PPV * TPR) / (PPV + TPR)
    //negative f1 score
    val F1N = (2 * NPV * TNR) / (NPV + TNR)

    (ACC, PPV, NPV, TPR, TNR, F1P, F1N, totalNum.toInt)
  }
}
