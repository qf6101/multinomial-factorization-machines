package io.github.qf6101.mfm.factorization

import io.github.qf6101.mfm.optimization.SquaredL2Updater
import io.github.qf6101.mfm.tuning.BinaryClassificationMetrics
import io.github.qf6101.mfm.util.{LoadDSUtil, MLlibTestSparkContext}
import org.apache.spark.ml.param.ParamMap
import org.joda.time.DateTime
import org.scalatest.FunSuite

/**
  * User: qfeng
  * Date: 15-12-8 下午4:58
  * Usage: test fm without cross validation
  */
class FmWithoutCV extends FunSuite with MLlibTestSparkContext {
  test("test fm without cross validation") {
    val (dataSet, numAttrs) = LoadDSUtil.loadLibSVMDataSet(System.getProperty("user.dir") + "/../testdata/mlalgorithms/input/a1a/a1a")
    val Array(training, testing) = dataSet.randomSplit(Array(0.9, 0.1))

    val paramPool = new ParamMap()
    val updater = new SquaredL2Updater()
    val fmLearn = new FmLearnSGD(paramPool, updater)
    paramPool.put(fmLearn.gd.numIterations, 10)
    paramPool.put(fmLearn.gd.stepSize, 0.3)
    paramPool.put(fmLearn.gd.miniBatchFraction, 1.0)
    paramPool.put(fmLearn.gd.convergenceTol, 1E-5)
    paramPool.put(fmLearn.numAttrs, numAttrs)
    paramPool.put(fmLearn.numFactors, 5)
    paramPool.put(fmLearn.k0, true)
    paramPool.put(fmLearn.k1, true)
    paramPool.put(fmLearn.k2, true)
    paramPool.put(fmLearn.maxInteractAttr, 2)
    paramPool.put(fmLearn.initMean, 0.0)
    paramPool.put(fmLearn.initStdev, 0.01)
    paramPool.put(fmLearn.reg0, 0.1)
    paramPool.put(fmLearn.reg1, 0.1)
    paramPool.put(fmLearn.reg2, 0.1)

    val model = fmLearn.train(training)
    val validating = testing.map { case (label, features) =>
      (model.regressionPredict(features), label)
    }
    val metrics = new BinaryClassificationMetrics(validating)
    val AUC = metrics.AUC
    println(AUC)
    model.saveModel(metrics.toString, System.getProperty("user.dir") + "/../testdata/mlalgorithms/output/" + DateTime.now().toString("yyyyMMdd.HHmmss"))
  }
}
