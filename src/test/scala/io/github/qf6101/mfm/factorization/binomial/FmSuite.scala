package io.github.qf6101.mfm.factorization.binomial

import io.github.qf6101.mfm.optimization.SquaredL2Updater
import io.github.qf6101.mfm.tuning.BinaryClassificationMetrics
import io.github.qf6101.mfm.util.TestingUtils._
import io.github.qf6101.mfm.util.{HDFSUtil, LoadDSUtil, MfmTestSparkSession}
import org.apache.spark.ml.param.ParamMap
import org.scalatest.FunSuite

/**
  * User: qfeng
  * Date: 15-12-8 下午4:58
  */
class FmSuite extends FunSuite with MfmTestSparkSession {
  test("test binomial factorization machines") {
    // Load training and testing data sets
    val (training, _) = LoadDSUtil.loadLibSVMDataSet("test_data/input/a1a/a1a")
    val (testing, numFeatures) = LoadDSUtil.loadLibSVMDataSet("test_data/input/a1a/a1a.t")
    // Construct factorization machines learner with parameters
    val params = new ParamMap()
    val updater = new SquaredL2Updater()
    val fmLearn = new FmLearnSGD(params, updater)
    params.put(fmLearn.gd.numIterations, 100)
    params.put(fmLearn.gd.stepSize, 0.05)
    params.put(fmLearn.gd.miniBatchFraction, 1.0)
    params.put(fmLearn.gd.convergenceTol, 1E-5)
    params.put(fmLearn.numFeatures, numFeatures)
    params.put(fmLearn.numFactors, 5)
    params.put(fmLearn.k0, true)
    params.put(fmLearn.k1, true)
    params.put(fmLearn.k2, true)
    params.put(fmLearn.maxInteractFeatures, numFeatures)
    params.put(fmLearn.initMean, 0.0)
    params.put(fmLearn.initStdev, 0.01)
    params.put(fmLearn.reg0, 0.01)
    params.put(fmLearn.reg1, 0.01)
    params.put(fmLearn.reg2, 0.01)
    // Train FM model
    val model = fmLearn.train(training)
    // Use testing data set to evaluate the model
    val eval = testing.map { case (label, features) =>
      (model.predict(features), label)
    }
    val metrics = new BinaryClassificationMetrics(eval)
    // Save model to file
    HDFSUtil.deleteIfExists("test_data/output/a1a")
    model.save("test_data/output/a1a")
    // Reload model from file and test if it is equal from the original model
    val reloadModel = FmModel("test_data/output/a1a")
    assert(model.equals(reloadModel))
    // Evaluate the reloaded model
    val reloadEval = testing.map { case (label, features) =>
      (model.predict(features), label)
    }
    // Test if the reloaded model has the same result on the testing data set
    val reloadMetrics = new BinaryClassificationMetrics(reloadEval)
    assert(reloadMetrics.AUC ~= metrics.AUC absTol 1E-5)
    // print the AUC
    println("AUC: " + metrics.AUC)
  }
}
