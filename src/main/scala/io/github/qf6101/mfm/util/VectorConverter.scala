package io.github.qf6101.mfm.util

import breeze.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vector

/**
  * Created by qfeng on 15-3-17.
  */
object VectorConverter {
  /**
    * spark的向量转成breeze的稀疏向量
    *
    * @param input spark向量
    * @return breeze的稀疏向量
    */
  def SparkVector2SV(input: Vector): SparseVector[Double] = {
    val result = SparseVector.zeros[Double](input.size)

    for (i <- 0 until input.size) {
      if (input(i) != 0.0) {
        result.update(i, input(i))
      }
    }

    result
  }
}
