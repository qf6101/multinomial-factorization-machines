package io.github.qf6101.mfm.factorization

import io.github.qf6101.mfm.factorization.binomial.FmModel
import org.apache.spark.{SparkConf, SparkContext}

/**
  * User: qfeng
  * Date: 16-1-18 上午11:12
  * Usage: 
  */
object FmModelTest {
  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName(FmModelTest.getClass.toString).setMaster("local"))
    val model = FmModel(System.getProperty("user.dir") + "/../testdata/mlalgorithms/output/20160113.191631")
  }
}
