package io.github.qf6101.mfm.optimization

import io.github.qf6101.mfm.logisticregression.{LogisticGradient, LrLearnLBFGS, VectorCoefficients}
import io.github.qf6101.mfm.util.MLlibTestSparkContext
import io.github.qf6101.mfm.util.TestingUtils._
import org.apache.spark.ml.param.ParamMap
import org.scalatest.FunSuite


/**
  * Created by qfeng on 15-4-7.
  */


class LBFGSSuite extends FunSuite with MLlibTestSparkContext {
  lazy val dataRDD = sc.parallelize(testData, 2).cache()
  val nPoints = 1000
  val A = 2.0
  val B = -1.5
  // Add a extra variable consisting of all 1.0's for the intercept.
  val testData = GradientDescentSuite.generateGDInput(A, B, nPoints, 42)
  val simpleUpdater = new SimpleUpdater()
  val squaredL2Updater = new SquaredL2Updater()

  test("LBFGS loss should be decreasing and match the result of Gradient Descent.") {
    val initialWeightsWithIntercept = new VectorCoefficients(2)
    initialWeightsWithIntercept.w.update(0, 1.0)
    initialWeightsWithIntercept.w.update(1, -1.0)

    val lbfgsParamPool = new ParamMap()
    val lbfgsGradient = new LogisticGradient(lbfgsParamPool)
    val lbfgsLrf = new LrLearnLBFGS(lbfgsParamPool, null)
    val lbfgs = new LBFGS(lbfgsGradient, simpleUpdater, lbfgsParamPool)

    lbfgsParamPool.put(lbfgs.numIterations, 10)
    lbfgsParamPool.put(lbfgsLrf.reg, Array(0.0))
    lbfgsParamPool.put(lbfgs.convergenceTol, 1e-12)
    lbfgsParamPool.put(lbfgs.numCorrections, 10)

    val (_, lossLBFGS) = lbfgs.optimizeWithHistory(
      dataRDD,
      initialWeightsWithIntercept,
      lbfgsParamPool(lbfgsLrf.reg))

    // Since the cost function is convex, the loss is guaranteed to be monotonically decreasing
    // with L-BFGS optimizer.
    // (SGD doesn't guarantee this, and the loss will be fluctuating in the optimization process.)
    assert((lossLBFGS, lossLBFGS.tail).zipped.forall(_ > _), "loss should be monotonically decreasing.")

    val gdParamPool = new ParamMap()
    val gdGradient = new LogisticGradient(gdParamPool)
    val gdLrf = new LrLearnLBFGS(lbfgsParamPool, null)
    val gd = new GradientDescent(gdGradient, simpleUpdater, gdParamPool)

    gdParamPool.put(gd.stepSize, 1.0)
    gdParamPool.put(gd.numIterations, 50)
    gdParamPool.put(gdLrf.reg, Array(0.0))
    gdParamPool.put(gd.miniBatchFraction, 1.0)
    gdParamPool.put(gd.convergenceTol, 1E-12)

    val (_, lossGD) = gd.optimizeWithHistory(
      dataRDD,
      initialWeightsWithIntercept,
      gdParamPool(gdLrf.reg))

    // GD converges a way slower than L-BFGS. To achieve 1% difference,
    // it requires 90 iterations in GD. No matter how hard we increase
    // the number of iterations in GD here, the lossGD will be always
    // larger than lossLBFGS. This is based on observation, no theoretically guaranteed
    assert(Math.abs((lossGD.last - lossLBFGS.last) / lossLBFGS.last) < 0.02,
      "LBFGS should match GD result within 2% difference.")
  }

  test("LBFGS and Gradient Descent with L2 regularization should get the same result.") {
    val initialWeightsWithIntercept = new VectorCoefficients(2)
    initialWeightsWithIntercept.w.update(0, 0.3)
    initialWeightsWithIntercept.w.update(1, 0.12)

    val lbfgsParamPool = new ParamMap()
    val lbfgsGradient = new LogisticGradient(lbfgsParamPool)
    val lbfgsLrf = new LrLearnLBFGS(lbfgsParamPool, null)
    val lbfgs = new LBFGS(lbfgsGradient, squaredL2Updater, lbfgsParamPool)

    lbfgsParamPool.put(lbfgs.numIterations, 10)
    lbfgsParamPool.put(lbfgsLrf.reg, Array(0.2))
    lbfgsParamPool.put(lbfgs.convergenceTol, 1e-12)
    lbfgsParamPool.put(lbfgs.numCorrections, 10)

    val (weightLBFGS, lossLBFGS) = lbfgs.optimizeWithHistory(
      dataRDD,
      initialWeightsWithIntercept,
      lbfgsParamPool(lbfgsLrf.reg))

    // Since the cost function is convex, the loss is guaranteed to be monotonically decreasing
    // with L-BFGS optimizer.
    // (SGD doesn't guarantee this, and the loss will be fluctuating in the optimization process.)
    assert((lossLBFGS, lossLBFGS.tail).zipped.forall(_ > _), "loss should be monotonically decreasing.")

    val gdParamPool = new ParamMap()
    val gdGradient = new LogisticGradient(gdParamPool)
    val gdLrf = new LrLearnLBFGS(lbfgsParamPool, null)
    val gd = new GradientDescent(gdGradient, squaredL2Updater, gdParamPool)

    gdParamPool.put(gd.stepSize, 1.0)
    gdParamPool.put(gd.numIterations, 50)
    gdParamPool.put(gdLrf.reg, Array(0.2))
    gdParamPool.put(gd.miniBatchFraction, 1.0)
    gdParamPool.put(gd.convergenceTol, 1E-12)

    val (weightGD, lossGD) = gd.optimizeWithHistory(
      dataRDD,
      initialWeightsWithIntercept,
      gdParamPool(gdLrf.reg))

    assert(lossGD(0) ~= lossLBFGS(0) absTol 1E-5,
      "The first losses of LBFGS and GD should be the same.")

    // The 2% difference here is based on observation, but is not theoretically guaranteed.
    assert(lossGD.last ~= lossLBFGS.last relTol 0.03,
      "The last losses of LBFGS and GD should be within 3% difference.")

    assert(
      (weightLBFGS.asInstanceOf[VectorCoefficients].w0 ~= weightGD.asInstanceOf[VectorCoefficients].w0 relTol 0.03)
      && (weightLBFGS.asInstanceOf[VectorCoefficients].w(0) ~= weightGD.asInstanceOf[VectorCoefficients].w(0) relTol 0.03)
        && (weightLBFGS.asInstanceOf[VectorCoefficients].w(1) ~= weightGD.asInstanceOf[VectorCoefficients].w(1) relTol 0.03),
      "The weight differences between LBFGS and GD should be within 3%.")
  }


  test("The convergence criteria should work as we expect.") {
    val initialWeightsWithIntercept = new VectorCoefficients(2)
    initialWeightsWithIntercept.w.update(0, 0.0)
    initialWeightsWithIntercept.w.update(1, 0.0)

    val lbfgsParamPool = new ParamMap()
    val lbfgsGradient = new LogisticGradient(lbfgsParamPool)
    val lbfgsLrf = new LrLearnLBFGS(lbfgsParamPool, null)
    val lbfgs = new LBFGS(lbfgsGradient, squaredL2Updater, lbfgsParamPool)

    lbfgsParamPool.put(lbfgs.numIterations, 8)
    lbfgsParamPool.put(lbfgsLrf.reg, Array(0.0))
    lbfgsParamPool.put(lbfgs.convergenceTol, 1E-12)
    lbfgsParamPool.put(lbfgs.numCorrections, 10)

    val (_, lossLBFGS1) = lbfgs.optimizeWithHistory(
      dataRDD,
      initialWeightsWithIntercept,
      lbfgsParamPool(lbfgsLrf.reg))

    // Note that the first loss is computed with initial weights,
    // so the total numbers of loss will be numbers of iterations + 1
    assert(lossLBFGS1.length == 9)

    lbfgsParamPool.put(lbfgs.convergenceTol, 0.1)

    val (_, lossLBFGS2) = lbfgs.optimizeWithHistory(
      dataRDD,
      initialWeightsWithIntercept,
      lbfgsParamPool(lbfgsLrf.reg))

    // Based on observation, lossLBFGS2 runs 3 iterations, no theoretically guaranteed.
    assert(lossLBFGS2.length == 4)
    assert((lossLBFGS2(2) - lossLBFGS2(3)) / lossLBFGS2(2) < 0.1)

    lbfgsParamPool.put(lbfgs.convergenceTol, 0.01)

    val (_, lossLBFGS3) = lbfgs.optimizeWithHistory(
      dataRDD,
      initialWeightsWithIntercept,
      lbfgsParamPool(lbfgsLrf.reg))

    // With smaller convergenceTol, it takes more steps.
    assert(lossLBFGS3.length > lossLBFGS2.length)

    // Based on observation, lossLBFGS2 runs 5 iterations, no theoretically guaranteed.
    assert(lossLBFGS3.length == 6)
    assert((lossLBFGS3(4) - lossLBFGS3(5)) / lossLBFGS3(4) < 0.01)
  }

}
