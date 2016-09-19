package io.github.qf6101.mfm.factorization.multinomial

import breeze.linalg.argmax
import io.github.qf6101.mfm.factorization.binomial.FmModel
import io.github.qf6101.mfm.optimization.SquaredL2Updater
import io.github.qf6101.mfm.util.TestingUtils._
import io.github.qf6101.mfm.util.{HDFSUtil, LoadDSUtil, MfmTestSparkSession}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.scalatest.FunSuite

/**
  * User: qfeng
  * Date: 15-12-8 下午4:58
  */
class MfmSuite extends FunSuite with MfmTestSparkSession {
  test("test binomial factorization machines") {
    val (training, numFeatures) = LoadDSUtil.loadLibSVMDataSet("test_data/input/mnist/mnist.scale")
    val (testing, _) = LoadDSUtil.loadLibSVMDataSet("test_data/input/mnist/mnist.scale.t")
    val params = new ParamMap()
    val updater = new SquaredL2Updater()
    val mfmLearn = new MfmLearnSGD(params, updater)
    params.put(mfmLearn.gd.numIterations, 10)
    params.put(mfmLearn.gd.stepSize, 0.5)
    params.put(mfmLearn.gd.miniBatchFraction, 1.0)
    params.put(mfmLearn.gd.convergenceTol, 1E-5)
    params.put(mfmLearn.numFeatures, numFeatures)
    params.put(mfmLearn.numFactors, 5)
    params.put(mfmLearn.k0, false)
    params.put(mfmLearn.k1, true)
    params.put(mfmLearn.k2, false)
    params.put(mfmLearn.maxInteractFeatures, numFeatures)
    params.put(mfmLearn.initMean, 0.0)
    params.put(mfmLearn.initStdev, 0.01)
    params.put(mfmLearn.reg0, 0.01)
    params.put(mfmLearn.reg1, 0.01)
    params.put(mfmLearn.reg2, 0.01)
    params.put(mfmLearn.numClasses, 10)
    val model = mfmLearn.train(training)
    val validating = testing.map { case (label, features) =>
      argmax(model.predict(features)).toDouble -> label
    }
    val metrics = new MulticlassMetrics(validating)
    println("accuracy: " + metrics.accuracy)
    println("weighted precision: " + metrics.weightedPrecision)
    println("weighted recall: " + metrics.weightedRecall)
    println("weighted f-measure: " + metrics.weightedFMeasure)
    HDFSUtil.deleteIfExists("test_data/output/mnist")
    model.save("test_data/output/mnist")

    val reloadModel = MfmModel("test_data/output/mnist")
    assert(model.equals(reloadModel))

    val reloadValidating = testing.map { case (label, features) =>
      argmax(reloadModel.predict(features)).toDouble -> label
    }
    val reloadMetrics = new MulticlassMetrics(reloadValidating)
    assert(reloadMetrics.accuracy ~= metrics.accuracy absTol 1E-5)
    assert(reloadMetrics.weightedPrecision ~= metrics.weightedPrecision absTol 1E-5)
    assert(reloadMetrics.weightedRecall ~= metrics.weightedRecall absTol 1E-5)
    assert(reloadMetrics.weightedFMeasure ~= metrics.weightedFMeasure absTol 1E-5)
  }
}
