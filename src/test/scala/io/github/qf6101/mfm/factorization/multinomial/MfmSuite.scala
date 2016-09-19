package io.github.qf6101.mfm.factorization.multinomial

import breeze.linalg.argmax
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
    // Load training and testing data sets
    val (training, numFeatures) = LoadDSUtil.loadLibSVMDataSet("test_data/input/mnist/mnist.scale")
    val (testing, _) = LoadDSUtil.loadLibSVMDataSet("test_data/input/mnist/mnist.scale.t")
    // Construct multinomial factorization machines learner with parameters
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
    // Train MFM model
    val model = mfmLearn.train(training)
    // Use testing data set to evaluate the model
    val eval = testing.map { case (label, features) =>
      argmax(model.predict(features)).toDouble -> label
    }
    val metrics = new MulticlassMetrics(eval)
    // Save model to file
    HDFSUtil.deleteIfExists("test_data/output/mnist")
    model.save("test_data/output/mnist")
    // Reload model from file and test if it is equal to the original model
    val reloadModel = MfmModel("test_data/output/mnist")
    assert(model.equals(reloadModel))
    // Evaluate the reloaded model
    val reloadEval = testing.map { case (label, features) =>
      argmax(reloadModel.predict(features)).toDouble -> label
    }
    // Test if the reloaded model has the same result on the testing data set
    val reloadMetrics = new MulticlassMetrics(reloadEval)
    assert(reloadMetrics.accuracy ~= metrics.accuracy absTol 1E-5)
    assert(reloadMetrics.weightedPrecision ~= metrics.weightedPrecision absTol 1E-5)
    assert(reloadMetrics.weightedRecall ~= metrics.weightedRecall absTol 1E-5)
    assert(reloadMetrics.weightedFMeasure ~= metrics.weightedFMeasure absTol 1E-5)
    // print the metrics
    println("accuracy: " + metrics.accuracy)
    println("weighted precision: " + metrics.weightedPrecision)
    println("weighted recall: " + metrics.weightedRecall)
    println("weighted f-measure: " + metrics.weightedFMeasure)
  }
}
