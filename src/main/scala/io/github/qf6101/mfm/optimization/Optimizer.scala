package io.github.qf6101.mfm.optimization

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.Coefficients
import org.apache.spark.rdd.RDD

/**
  * Created by qfeng on 15-3-11.
  */
trait Optimizer extends Serializable {
  def optimize(data: RDD[(Double, SparseVector[Double])],
               initialCoeffs: Coefficients,
               regParam: Array[Double]): Coefficients
}
