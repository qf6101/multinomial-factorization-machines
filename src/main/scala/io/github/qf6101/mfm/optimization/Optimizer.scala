package io.github.qf6101.mfm.optimization

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.Coefficients
import org.apache.spark.rdd.RDD

/**
  * Created by qfeng on 15-3-11.
  */

/**
  * 优化器接口,实现包括SGD, LBFGS等
  */
trait Optimizer extends Serializable {
  /**
    * 最优化函数
    *
    * @param data          样本数据集
    * @param initialCoeffs 初始化系数值
    * @param regParam      正则参数值
    * @return 学习后的参数
    */
  def optimize(data: RDD[(Double, SparseVector[Double])],
               initialCoeffs: Coefficients,
               regParam: Array[Double]): Coefficients
}
