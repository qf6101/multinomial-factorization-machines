package io.github.qf6101.mfm.util

import org.apache.spark.ml.param.{Param, ParamMap}
import org.scalatest.FunSuite

/**
  * User: qfeng
  * Date: 15-8-11 下午4:44
  * Usage:
  */
class ParamSuite extends FunSuite {
  test("add two parameter sets") {
    val params = new ParamMap()
    val otherParams = new ParamMap()

    val param1: Param[Double] = new Param("ParamTest", "param1", "param1")
    val param2: Param[Double] = new Param("ParamTest", "param2", "param2")
    val param3: Param[Double] = new Param("ParamTest", "param3", "param3")
    val param4: Param[Double] = new Param("ParamTest", "param4", "param4")

    params.put[Double](param1, 1.0)
    params.put[Double](param1, 2.0)
    params.put[Double](param2, 5.0)

    otherParams.put[Double](param2, 10.1)
    otherParams.put[Double](param3, 7.0)
    otherParams.put[Double](param4, 8.0)

    params ++= otherParams

    assert(params(param1) == 2.0)
    //overwrite by other parameters
    assert(params(param2) == 10.1)
    assert(params(param3) == 7.0)
    assert(params(param4) == 8.0)
    //print parameters
    println(ParamUtil.paramsToString(params))
  }
}
