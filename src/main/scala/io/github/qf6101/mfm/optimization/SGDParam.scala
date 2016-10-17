package io.github.qf6101.mfm.optimization

import org.apache.spark.ml.param.{Param, ParamValidators}


/**
  * Created by qfeng on 15-3-18.
  */


/**
  * SGD的参数
  */
trait SGDParam extends Serializable {
  //default value: 1.0
  val stepSize: Param[Double] = new Param("SGDParam", "stepSize", "initial step size for the first step",
    ParamValidators.gt(0))
  val numIterations: Param[Int] = new Param("SGDParam", "numIterations", "number of iterations that SGD should be run",
    ParamValidators.gt(0))
  //default value: 1.0
  val miniBatchFraction: Param[Double] = new Param("SGDParam", "miniBatchFraction", "fraction of the input data set " +
    "that should be used for one iteration of SGD", ParamValidators.inRange(0, 1, false, true))
  //default value:1E-4
  val convergenceTol: Param[Double] = new Param("SGDParam", "convergenceTol", "convergence tolerance of iterations for SGD",
    ParamValidators.gt(0))
}

