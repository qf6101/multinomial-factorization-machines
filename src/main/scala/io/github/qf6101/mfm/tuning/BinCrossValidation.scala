package io.github.qf6101.mfm.tuning

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.binary.{BinLearner, BinModel}
import io.github.qf6101.mfm.baseframe.{MLLearner, MLModel}
import io.github.qf6101.mfm.util.{Logging, ParamUtil}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
  * User: qfeng
  * Date: 15-8-24 上午10:25
  * Usage: 快速模型选择，固定其他参数，尝试某一参数的各个值，取最大AUC的参数值
  */

class BinCrossValidation(val learner: BinLearner,
                         val paramGridBuilder: BinParamGridBuilder,
                         val numFolds: Int = 5,
                         val baseParamMinAUC: Double = 0.0) extends Logging with Serializable {

  /**
    * 分类问题的模型选择
    *
    * @param dataset 数据集
    * @return 训练得到的模型及其评估值
    */
  def selectParamsForClassif(dataset: RDD[(Double, SparseVector[Double])]): (BinModel, BinaryClassificationMetrics)
  = {
    //选择得到的参数集合，即返回值
    val selectedParamMap = new ParamMap
    //数据分块，用于交叉验证
    val splits = MLUtils.kFold(dataset, numFolds, Random.nextInt())
    //随机选择一组参数作为基准参数
    val baseParamMap = selectBaseParams(splits(0)._1, baseParamMinAUC)
    //对于每个参数，都尝试它的每个参数值，选择AUC最大的那个作为最终的参数值（其他参数采用基准参数值）
    paramGridBuilder.paramGrid.foreach { case (param, paramValues) =>
      //每个参数值都对应数组中的一个元素
      val AUCs = new Array[Double](paramValues.size)
      val models = new Array[BinModel](paramValues.size)
      val candidateParamValues = new Array[Any](paramValues.size)

      //对于每个参数值，都基于交叉检验训练模型，计算AUC均值
      paramValues.zipWithIndex.foreach { case (paramValue, paramValueIndex) =>
        //组装出一组参数，用于训练模型
        val paramMap = baseParamMap.copy.put(param.asInstanceOf[Param[Any]], paramValue)
        candidateParamValues(paramValueIndex) = paramValue
        learner.updateParams(paramMap)
        //采用交叉检验训练模型计算AUC值
        splits.zipWithIndex.foreach { case ((training, testing), splitIndex) =>
          models(paramValueIndex) = learner.train(training)
          val validating = testing.map { case (label, features) =>
            (models(paramValueIndex).regressionPredict(features), label)
          }
          val metrics = new BinaryClassificationMetrics(validating)
          val AUC = metrics.AUC
          AUCs(paramValueIndex) += AUC
          logInfo(s"split $splitIndex >>>>> AUC: ${metrics.AUC}")
        }
        //计算AUC均值
        AUCs(paramValueIndex) /= splits.length
        logInfo(s"selected parameters: ${ParamUtil.paramPoolToString(paramMap)}; >>>>> AUC: ${AUCs(paramValueIndex).formatted("%1.4f")}")
      }
      //挑选出AUC最大的参数值
      val (_, bestIndex) = AUCs.zipWithIndex.maxBy(_._1)
      selectedParamMap.put(param.asInstanceOf[Param[Any]], candidateParamValues(bestIndex))
    }
    //使用挑选出的那组参数值，基于整个数据集训练模型，并计算评估值
    learner.updateParams(selectedParamMap)
    val fullModel = learner.train(dataset)
    val fullValidating = dataset.map { case (label, features) =>
      (fullModel.regressionPredict(features), label)
    }
    val fullMetrics = new BinaryClassificationMetrics(fullValidating)
    (fullModel, fullMetrics)
  }

  /**
    * 随机选择一组参数作为基准参数
    *
    * @param dataset 数据集
    * @param baseParamMinAUC 基准参数的AUC阈值（基准参数描述的模型AUC不能小于等于该阈值）
    * @return 基准参数
    */
  private def selectBaseParams(dataset: RDD[(Double, SparseVector[Double])],
                               baseParamMinAUC: Double): ParamMap = {
    var selected = false
    var baseParamMap: ParamMap = null
    var tryTime: Int = 0

    while (!selected) {
      //尝试5次，如果AUC都是0则抛出异常
      tryTime = tryTime + 1
      if (tryTime > 5) {
        throw new Exception("try time exceeds 5 for base parameters selection.")
      }
      //随机选择一组参数，并计算AUC值
      baseParamMap = paramGridBuilder.sampleParams()
      learner.updateParams(baseParamMap)
      val model = learner.train(dataset)
      val validating = dataset.map { case (label, features) =>
        (model.regressionPredict(features), label)
      }
      val metrics = new BinaryClassificationMetrics(validating)
      //AUC值大于阈值，则返回
      if (metrics.AUC > baseParamMinAUC) {
        selected = true
      }
    }
    baseParamMap
  }

}
