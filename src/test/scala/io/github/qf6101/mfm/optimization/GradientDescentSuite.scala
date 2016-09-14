package io.github.qf6101.mfm.optimization

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.logisticregression.{LogisticGradient, LrLearnSGD, VectorCoefficients}
import io.github.qf6101.mfm.util.MfmTestSparkSession
import io.github.qf6101.mfm.util.TestingUtils._
import org.apache.spark.ml.param.ParamMap
import org.scalatest.{FunSuite, Matchers}

import scala.collection.JavaConversions._
import scala.util.Random

/**
  * Created by qfeng on 15-3-13.
  */

object GradientDescentSuite {

  def generateLogisticInputAsList(
                                   offset: Double,
                                   scale: Double,
                                   nPoints: Int,
                                   seed: Int): java.util.List[(Double, SparseVector[Double])] = {
    seqAsJavaList(generateGDInput(offset, scale, nPoints, seed))
  }

  // Generate input of the form Y = logistic(offset + scale * X)
  def generateGDInput(offset: Double,
                      scale: Double,
                      nPoints: Int,
                      seed: Int): Seq[(Double, SparseVector[Double])] = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Double](nPoints)(rnd.nextGaussian())

    val unifRand = new Random(45)
    val rLogis = (0 until nPoints).map { i =>
      val u = unifRand.nextDouble()
      math.log(u) - math.log(1.0 - u)
    }

    val y: Seq[Double] = (0 until nPoints).map { i =>
      val yVal = offset + scale * x1(i) + rLogis(i)
      if (yVal > 0) 1.0 else 0.0
    }

    (0 until nPoints).map(i => (y(i), new SparseVector[Double](Array(0, 1), Array(1.0, x1(i)), 2)))
  }
}

class GradientDescentSuite extends FunSuite with MfmTestSparkSession with Matchers {
  test("Assert the loss is decreasing.") {
    val nPoints = 1000
    val A = 2.0
    val B = -1.5

    val params = new ParamMap()
    val gradient = new LogisticGradient(params)
    val updater = new SimpleUpdater()
    val lrf = new LrLearnSGD(params, null)
    val gd = new GradientDescent(gradient, updater, params)

    // Add a extra variable consisting of all 1.0's for the intercept.
    val testData = GradientDescentSuite.generateGDInput(A, B, nPoints, 42)
    val dataRDD = spark.sparkContext.parallelize(testData, 2).cache()
    val initialWeightsWithIntercept = new VectorCoefficients(2)
    initialWeightsWithIntercept.w.update(0, 1.0)
    initialWeightsWithIntercept.w.update(1, -1.0)

    params.put(gd.numIterations, 10)
    params.put(gd.miniBatchFraction, 1.0)
    params.put(gd.stepSize, 1.0)
    params.put(gd.convergenceTol, 1E-4)
    params.put(lrf.reg, Array(0.0))

    val (_, loss) = gd.optimizeWithHistory(
      dataRDD,
      initialWeightsWithIntercept,
      params(lrf.reg))

    assert(loss.last - loss.head < 0, "loss isn't decreasing.")

    val lossDiff = loss.init.zip(loss.tail).map { case (lhs, rhs) => lhs - rhs }
    assert(lossDiff.count(_ > 0).toDouble / lossDiff.size > 0.8)
  }


  test("Test the loss and gradient of first iteration with regularization.") {
    val params = new ParamMap()
    val gradient = new LogisticGradient(params)
    val updater = new SquaredL2Updater()
    val lrf = new LrLearnSGD(params, null)
    val gd = new GradientDescent(gradient, updater, params)

    // Add a extra variable consisting of all 1.0's for the intercept.
    val testData = GradientDescentSuite.generateGDInput(2.0, -1.5, 1000, 42)
    val dataRDD = spark.sparkContext.parallelize(testData, 2).cache()

    // Prepare non-zero weights
    val initialWeightsWithIntercept = new VectorCoefficients(2)
    initialWeightsWithIntercept.w.update(0, 1.0)
    initialWeightsWithIntercept.w.update(1, 0.5)

    params.put(gd.numIterations, 1)
    params.put(gd.miniBatchFraction, 1.0)
    params.put(gd.stepSize, 1.0)
    params.put(gd.convergenceTol, 1E-4)
    params.put(lrf.reg, Array(0.0))

    val (newWeights0, loss0) = gd.optimizeWithHistory(
      dataRDD, initialWeightsWithIntercept, params(lrf.reg))

    params.put(gd.numIterations, 1)
    params.put(lrf.reg, Array(1.0))

    val (newWeights1, loss1) = gd.optimizeWithHistory(
      dataRDD, initialWeightsWithIntercept, params(lrf.reg))

    assert(
      loss1(0) ~= (loss0(0) + (math.pow(initialWeightsWithIntercept.w(0), 2) +
        math.pow(initialWeightsWithIntercept.w(1), 2)) / 2) absTol 1E-5,
      """For non-zero weights, the regVal should be \frac{1}{2}\sum_i w_i^2.""")

    assert(
      (newWeights1.asInstanceOf[VectorCoefficients].w(0) ~= (newWeights0.asInstanceOf[VectorCoefficients].w(0) -
        initialWeightsWithIntercept.w(0))
        absTol 1E-5) &&
        (newWeights1.asInstanceOf[VectorCoefficients].w(1) ~= (newWeights0.asInstanceOf[VectorCoefficients].w(1) -
          initialWeightsWithIntercept.w(1)) absTol 1E-5),
      "The different between newWeights with/without regularization " +
        "should be initialWeightsWithIntercept.")
  }
}
