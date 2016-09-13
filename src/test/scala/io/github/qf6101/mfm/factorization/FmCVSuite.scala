package io.github.qf6101.mfm.factorization

import io.github.qf6101.mfm.factorization.binomial.FmLearnSGD
import io.github.qf6101.mfm.optimization.SquaredL2Updater
import io.github.qf6101.mfm.tuning.{BinCrossValidation, BinParamGridBuilder}
import io.github.qf6101.mfm.util.{LoadDSUtil, MLlibTestSparkContext}
import org.apache.spark.ml.param.ParamMap
import org.joda.time.DateTime
import org.scalatest.FunSuite

/**
  * Created by qfeng on 15-4-2.
  */

class FmCVSuite extends FunSuite with MLlibTestSparkContext {
  test("test factorization machine with cross validation") {
    val params = new ParamMap()
    val updater = new SquaredL2Updater()
    val fmLearn = new FmLearnSGD(params, updater)

    val (rawDataset, numAttrs) = LoadDSUtil.loadLibSVMDataSet(System.getProperty("user.dir") + "/../testdata/mlalgorithms/input/a1a/a1a")
    val dataset = rawDataset.map { case (label, data) =>
      if (label > 0) (1.0, data) else (-1.0, data)
    }.repartition(10).cache()

    params.put(fmLearn.gd.numIterations, 10)
    params.put(fmLearn.gd.stepSize, 1.0)
    params.put(fmLearn.gd.miniBatchFraction, 1.0)
    params.put(fmLearn.gd.convergenceTol, 1E-5)
    params.put(fmLearn.numFeatures, numAttrs)
    params.put(fmLearn.numFactors, 5)
    params.put(fmLearn.k0, true)
    params.put(fmLearn.k1, true)
    params.put(fmLearn.k2, true)
    params.put(fmLearn.maxInteractFeatures, numAttrs)
    params.put(fmLearn.initMean, 0.0)
    params.put(fmLearn.initStdev, 0.1)
    params.put(fmLearn.reg0, 0.1)
    params.put(fmLearn.reg2, 0.1)

    val paramGrid = new BinParamGridBuilder()
      .addGrid(fmLearn.reg1, Array(0.1, 0.05))

    val cv = new BinCrossValidation(fmLearn, paramGrid, 3)
    val (model, metrics) = cv.selectParamsForClassif(dataset)
//    model.saveModel(metrics.toString, System.getProperty("user.dir") + "/../testdata/mlalgorithms/output/" + DateTime.now().toString("yyyyMMdd.HHmmss"))
  }
}
