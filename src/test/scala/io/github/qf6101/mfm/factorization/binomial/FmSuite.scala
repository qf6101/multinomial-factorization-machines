package io.github.qf6101.mfm.factorization.binomial

import io.github.qf6101.mfm.optimization.SquaredL2Updater
import io.github.qf6101.mfm.tuning.BinaryClassificationMetrics
import io.github.qf6101.mfm.util.{HDFSUtil, LoadDSUtil, MfmTestSparkSession}
import org.apache.spark.ml.param.ParamMap
import org.scalatest.FunSuite
import io.github.qf6101.mfm.util.TestingUtils._

/**
  * User: qfeng
  * Date: 15-12-8 下午4:58
  */
class FmSuite extends FunSuite with MfmTestSparkSession {
  test("test binomial factorization machines") {
    val (training, _) = LoadDSUtil.loadLibSVMDataSet("test_data/input/a1a/a1a")
    val (testing, numFeatures) = LoadDSUtil.loadLibSVMDataSet("test_data/input/a1a/a1a.t")
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
    val model = fmLearn.train(training)
    val validating = testing.map { case (label, features) =>
      (model.predict(features), label)
    }
    val metrics = new BinaryClassificationMetrics(validating)
    HDFSUtil.deleteIfExists("test_data/output/a1a")
    model.save("test_data/output/a1a")

    val reloadModel = FmModel("test_data/output/a1a")
    assert(model.equals(reloadModel))

    val reloadValidating = testing.map { case (label, features) =>
      (model.predict(features), label)
    }
    val reloadMetrics = new BinaryClassificationMetrics(reloadValidating)
    assert(reloadMetrics.AUC ~= metrics.AUC absTol 1E-5)
    println("AUC: " + metrics.AUC)
  }
}
