package io.github.qf6101.mfm.logisticregression

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.MLModel
import io.github.qf6101.mfm.baseframe.binary.BinModel
import io.github.qf6101.mfm.util.Logging
import org.apache.spark.SparkContext
import org.apache.spark.ml.param.ParamMap

/**
  * Created by qfeng on 15-3-16.
  */

/**
  * 逻辑斯蒂回归模型
 *
  * @param coeffs 模型系数
  * @param paramMeta 逻辑斯蒂参数
  * @param params 参数池
  */
class LrModel(override val coeffs: VectorCoefficients,
              override val paramMeta: LrModelParam,
              override val params: ParamMap)
  extends BinModel(coeffs, paramMeta, params) with Logging {
  /**
    * 对输入数据进行预测（使用内置系数）
 *
    * @param data 输入数据
    * @return 预测值(0~1)
    */
  override def regressionPredict(data: SparseVector[Double]): Double = {
    predict(data, this.coeffs)
  }

  /**
    * 对输入数据进行预测
 *
    * @param data 输入数据
    * @param coeffs 系数
    * @return 预测值(0~1)
    */
  def predict(data: SparseVector[Double], coeffs: VectorCoefficients = this.coeffs): Double = {
    val margin = -1.0 * coeffs.dot(data)
    1.0 / (1.0 + math.exp(margin))
  }
}

object LrModel extends Logging {
  /**
    * 从文件载入逻辑斯蒂模型
    *
    * @param file 包含逻辑斯蒂模型信息的文件(可以从LrModel.saveModel获得)
    * @return 逻辑斯蒂模型
    */
  def apply(file: String): LrModel = {
    apply(SparkContext.getOrCreate().textFile(file).collect())
  }

  /**
    * 从字符串数组载入逻辑斯蒂模型
    *
    * @param content 包含模型信息的字符串数组（可以从LrModel.toString.split("\n")获得）
    * @return 逻辑斯蒂模型
    */
  def apply(content: Array[String]): LrModel = {
    var currentLine = 0
    //    var featureMapper: Option[FeatureMapper] = None
    //解析特征映射(feature segment)
    //    val numFeatureSegmentLines = content(currentLine).split(":")(1).split(" ")(0).trim.toInt
    //    if (numFeatureSegmentLines > 0) {
    //      currentLine += 1
    //      val endLine = currentLine + numFeatureSegmentLines + 1
    //      val featureLines = content.slice(currentLine, endLine)
    //      currentLine = endLine
    //      featureMapper = Some(FeatureMapper(featureLines))
    //    }
    //解析模型系数(coefficient segment)
    val numCoeffSegmentLines = content(currentLine).split(":")(1).split(" ")(0).trim.toInt
    currentLine += 1
    val endLine = currentLine + numCoeffSegmentLines + 1
    val coefficientsLines = content.slice(currentLine, endLine)
    currentLine = endLine
    val coefficients = VectorCoefficients(coefficientsLines)
    //解析模型参数(parameter segment)
    val params = new ParamMap()
    val lrModelParam = LrModelParam(content(currentLine + 1), params)
    //返回结果
    new LrModel(coefficients, lrModelParam, params)
  }
}