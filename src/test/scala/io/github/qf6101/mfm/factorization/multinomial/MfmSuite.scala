package io.github.qf6101.mfm.factorization.multinomial

import io.github.qf6101.mfm.optimization.SquaredL2Updater
import io.github.qf6101.mfm.tuning.BinaryClassificationMetrics
import io.github.qf6101.mfm.util.{HDFSUtil, LoadDSUtil, MfmTestSparkSession}
import org.apache.spark.ml.param.ParamMap
import org.scalatest.FunSuite

/**
  * User: qfeng
  * Date: 15-12-8 下午4:58
  */
class MfmSuite extends FunSuite with MfmTestSparkSession {
  test("test binomial factorization machines") {
    val (training, numFeatures) = LoadDSUtil.loadLibSVMDataSet("test_data/input/mnist/mnist.scale")
//    val (testing, numFeatures) = LoadDSUtil.loadLibSVMDataSet("test_data/input/mnist/mnist.scale.t")
    val params = new ParamMap()
    val updater = new SquaredL2Updater()
    val mfmLearn = new MfmLearnSGD(params, updater)
    params.put(mfmLearn.gd.numIterations, 100)
    params.put(mfmLearn.gd.stepSize, 0.05)
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
//    val validating = testing.map { case (label, features) =>
//      (model.predict(features), label)
//    }
//    val metrics = new BinaryClassificationMetrics(validating)
//    val AUC = metrics.AUC
//    println(AUC)
//    HDFSUtil.deleteIfExists("test_data/output/a1a")
//    model.save("test_data/output/a1a")
  }
}
