package io.github.qf6101.mfm.util

import java.util.Random

import breeze.linalg.{DenseMatrix, DenseVector}


/**
  * Created by qfeng on 15-1-26.
  */

/**
  * 高斯随机数生成器实例
  */
object GaussianRandom {
  /**
    * 生成高斯随机数密向量
    * @param mean 高斯分布的均值
    * @param stdev 高斯分布的标准差
    * @param length 向量长度
    * @return 高斯随机数密向量
    */
  def randDenseVector(mean: Double, stdev: Double, length: Int): DenseVector[Double] = {
    val results = DenseVector.zeros[Double](length)
    for (i <- 0 until length) {
      results.update(i, rand(mean, stdev))
    }
    results
  }

  /**
    * 生成告诉随机数密矩阵
    * @param mean 高斯分布的均值
    * @param stdev 高斯分布的标准差
    * @param numRows 矩阵行数
    * @param numCols 矩阵列数
    * @return 高斯随机数密矩阵
    */
  def randDenseMatrix(mean: Double, stdev: Double, numRows: Int, numCols: Int): DenseMatrix[Double] = {
    val results = DenseMatrix.zeros[Double](numRows, numCols)
    for (i <- 0 until numRows)
      for (j <- 0 until numCols)
        results.update(i, j, rand(mean, stdev))
    results
  }

  /**
    * 生成高斯随机数
    * @param mean 高斯分布的均值
    * @param stdev 高斯分布的标准差
    * @return 高斯随机数
    */
  def rand(mean: Double, stdev: Double): Double = {
    val random = new Random()
    val genValue = random.nextGaussian()
    mean + stdev * genValue
  }
}
