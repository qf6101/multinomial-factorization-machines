package io.github.qf6101.mfm.factorization

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.base.MLModel
import io.github.qf6101.mfm.util.Logging
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap


/**
  * Created by qfeng on 15-1-26.
  */

/**
  * Factorization Machine模型
  * @param paramHolder 分解机模型参数
  * @param paramPool 参数池
  */
class FmModel(override val coeffs: FmCoefficients,
              override val paramHolder: FmModelParam,
              override val paramPool: ParamMap)
  extends MLModel(coeffs, paramHolder, paramPool) {
  /**
    * 对输入数据进行预测
    * @param data 输入数据
    * @return 预测值
    */
  override def regressionPredict(data: SparseVector[Double]): Double = {
    val score = FmModel.linearScore(data, paramHolder, paramPool, coeffs)
    1.0 / (1.0 + math.exp(-score))
  }
}

object FmModel extends Logging {
  def linearScore(data: SparseVector[Double], paramHolder: FmModelParam, paramPool: ParamMap, coeffs: FmCoefficients): Double = {
    //初始化各阶预测值为0
    var zeroWayPredict = 0.0
    var oneWayPredict = 0.0
    var twoWayPredict = 0.0
    //参与2阶项的最大维度
    val maxInteractAttr = paramPool(paramHolder.maxInteractAttr)
    //0阶预测值
    if (paramPool(paramHolder.k0)) {
      zeroWayPredict += coeffs.w0
    }
    //1阶预测值
    if (paramPool(paramHolder.k1)) {
      data.activeIterator.foreach { case (index, value) =>
        oneWayPredict += coeffs.w(index) * value
      }
    }
    //2阶预测值
    if (paramPool(paramHolder.k2)) {
      for (factorIndex <- 0 until paramPool(paramHolder.numFactors)) {
        var firstMoment = 0.0
        var secondMoment = 0.0
        data.activeIterator.foreach { case (index, value) =>
          if (index < maxInteractAttr) {
            firstMoment += coeffs.v(index, factorIndex) * value
            secondMoment += math.pow(coeffs.v(index, factorIndex) * value, 2)
          }
        }
        twoWayPredict += firstMoment * firstMoment - secondMoment
      }
    }
    zeroWayPredict + oneWayPredict + 0.5 * twoWayPredict
  }

  /**
    * 从文件载入分解机模型
    *
    * @param file 包含分解机型信息的文件(可以从FmModel.saveModel获得)
    * @return 分解机模型
    */
  def apply(file: String): FmModel = {
    FmModel(SparkContext.getOrCreate().textFile(file).collect())
  }

  /**
    * 从字符串数组载入分解机模型
    *
    * @param content 包含模型信息的字符串数组（可以从FmModel.toString.split("\n")获得）
    * @return 分解机模型
    */
  def apply(content: Array[String]): FmModel = {
    var currentLine = 0
    val numCoeffSegmentLines = content(currentLine).split(":")(1).split(" ")(1).trim.toInt
    currentLine += 1
    val endLine = currentLine + numCoeffSegmentLines
    val coefficientsLines = content.slice(currentLine, endLine)
    currentLine = endLine
    val coefficients = FmCoefficients(coefficientsLines)
    //解析模型参数(parameter segment)
    val paramPool = new ParamMap()
    val fmModelParam = FmModelParam(content(currentLine + 1), paramPool)
    //返回结果
    new FmModel(coefficients, fmModelParam, paramPool)
  }
}