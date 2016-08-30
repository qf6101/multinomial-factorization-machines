package io.github.qf6101.mfm.optimization

import org.apache.spark.ml.param.{Param, ParamValidators}

/**
  * Created by qfeng on 15-4-7.
  */

/**
  * LGBGS的参数
  */
trait LBFGSParam extends Serializable {
  //default value: 10
  val numCorrections: Param[Int] = new Param("LBFGSParam", "numCorrections", "number of corrections used in the LBFGS " +
    "update", ParamValidators.gt(0))
  //default value:1E-4
  val convergenceTol: Param[Double] = new Param("LBFGSParam", "convergenceTol", "convergence tolerance of iterations for LBFGS",
    ParamValidators.gt(0))
  val numIterations: Param[Int] = new Param("LBFGSParam", "numIterations", "number of iterations that SGD should be run",
    ParamValidators.gt(0))
}
