package io.github.qf6101.mfm.util

import org.apache.spark.ml.param.{Param, ParamMap}

/**
  * User: qfeng
  * Date: 15-8-11 下午4:44
  * Usage:
  */
object ParamTest {
  def main(args: Array[String]) {
    val paramPool = new ParamMap()
    val otherPool = new ParamMap()

    val param1: Param[Double] = new Param("ParamTest", "param1", "param1")
    val param2: Param[Double] = new Param("ParamTest", "param2", "param2")
    val param3: Param[Double] = new Param("ParamTest", "param3", "param3")
    val param4: Param[Double] = new Param("ParamTest", "param4", "param4")

    paramPool.put[Double](param1, 1.0)
    paramPool.put[Double](param1, 2.0)
    paramPool.put[Double](param2, 5.0)

    otherPool.put[Double](param2, 10.0)
    otherPool.put[Double](param3, 7.0)
    otherPool.put[Double](param4, 8.0)

    paramPool ++= otherPool
    println(ParamUtil.paramPoolToString(paramPool))
  }
}
