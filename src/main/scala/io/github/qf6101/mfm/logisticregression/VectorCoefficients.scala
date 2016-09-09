package io.github.qf6101.mfm.logisticregression

import breeze.linalg.SparseVector
import io.github.qf6101.mfm.baseframe.Coefficients

import scala.collection.mutable
import scala.collection.mutable.HashMap
import scala.math._

/**
  * Created by qfeng on 15-6-11.
  */
class VectorCoefficients(val size: Int) extends Coefficients {
  var w0 = 0.0
  var w = HashMap[Int, Double]()

  /**
    * 同时复制this的结构和内容
    * @return 复制的拷贝
    */
  override def copy: Coefficients = {
    new VectorCoefficients(this.size, this.w0, this.w)
  }

  /**
    * 用Map稀疏向量初始化
    * @param w0 截距
    * @param w Map稀疏向量表示的参数
    */
  def this(size: Int, w0: Double, w: HashMap[Int, Double]) {
    this(size)
    this.w0 = w0
    this.w ++= w
  }

  /**
    * 只复制this的结构（比如参数个数），不复制内容
    * @return 复制的拷贝
    */
  override def copyEmpty(): Coefficients = new VectorCoefficients(this.size)

  /**
    * 对应系数加法，加至this上
    * @param otherW0 截距加数
    * @param otherW 一阶系数加数
    * @return this
    */
  def +=(otherW0: Double, otherW: SparseVector[Double]): VectorCoefficients = {
    this.w0 += otherW0
    otherW.activeIterator.foreach { case (index, value) =>
      val originalValue = this.w.getOrElse(index, 0.0)
      this.w.update(index, originalValue + value)
    }
    this
  }

  /**
    * 对应系数加法，加至this上
    * @param other 加数
    * @return this
    */
  override def +=(other: Coefficients): Coefficients = {
    val otherCoeffs = other.asInstanceOf[VectorCoefficients]
    this.w0 += otherCoeffs.w0
    otherCoeffs.w.foreach { case (index, value) =>
      val originalValue = this.w.getOrElse(index, 0.0)
      this.w.update(index, originalValue + value)
    }
    this
  }

  /**
    * 对应系数减法，减至this上
    * @param other 减数
    * @return this
    */
  override def -=(other: Coefficients): Coefficients = {
    val otherCoeffs = other.asInstanceOf[VectorCoefficients]
    this.w0 -= otherCoeffs.w0
    otherCoeffs.w.foreach { case (index, value) =>
      val originalValue = this.w.getOrElse(index, 0.0)
      this.w.update(index, originalValue - value)
    }
    this
  }

  /**
    *
    * 对应系数加上同一实数，加至复制this的类上
    * @param addend 加数
    * @return 加法结果（拷贝）
    */
  override def +(addend: Double): Coefficients = {
    val result = new VectorCoefficients(this.size)
    result.w0 = this.w0 + addend
    result.w = this.w.map { case (index, value) => index -> (value + addend) }
    result
  }

  /**
    * 对应系数除上同一实数，加至复制this的类上
    * @param dividend 除数
    * @return 除法结果
    */
  override def /(dividend: Double): Coefficients = {
    val result = new VectorCoefficients(this.size)
    result.w0 = this.w0 / dividend
    result.w = this.w.map { case (index, value) => index -> (value / dividend) }
    result
  }

  /**
    * 计算L2的正则值
    * @param reg 正则参数
    * @return 参数加权后的L2正则值
    */
  override def L2RegValue(reg: Array[Double]): Double = {
    var squaredCoeffSum = w0 * w0
    this.w.foreach { case (index, value) =>
      squaredCoeffSum += value * value
    }
    0.5 * reg(0) * squaredCoeffSum
  }

  /**
    * 计算L2的正则梯度值
    * @param reg 正则参数
    * @return 参数加权后的L2正则梯度值
    */
  override def L2RegGradient(reg: Array[Double]): Coefficients = {
    this * reg(0)
  }

  /**
    * 对应系数乘上同一实数，加至复制this的类上
    * @param multiplier  乘数
    * @return 乘法结果
    */
  override def *(multiplier: Double): Coefficients = {
    val result = new VectorCoefficients(this.size)
    result.w0 = this.w0 * multiplier
    result.w = this.w.map { case (index, value) => index -> (value * multiplier) }
    result
  }

  /**
    * 用L1稀疏化系数
    *
    * @param regParam 正则参数值
    * @param stepSize 学习率
    * @return 稀疏化后的系数
    */
  override def L1Shrink(regParam: Array[Double], stepSize: Double): Coefficients = {
    //收缩值
    val shrinkageVal = regParam(0) * stepSize
    w0 = signum(w0) * max(0.0, abs(w0) - shrinkageVal)
    w = w.flatMap { case (index, weight) =>
      val newWeight = signum(weight) * max(0.0, abs(weight) - shrinkageVal)
      if (newWeight == 0) {
        Nil
      } else {
        List(index -> newWeight)
      }
    }
    this
  }

  /**
    * 计算L1的正则值
    * @param regParam 正则参数
    * @return 参数绝对值加权后的L1正则值
    */
  override def L1RegValue(regParam: Array[Double]): Double = {
    val zeroRegValue = abs(w0)
    val firstRegValue = this.w.foldLeft(0.0) { case (absSum, element) =>
      absSum + abs(element._2)
    }
    (zeroRegValue + firstRegValue) * regParam(0)
  }

  /**
    * 系数与稀疏向量点乘
    *
    * @param otherW 稀疏向量
    * @return 点乘的结果
    */
  def dot(otherW: SparseVector[Double]): Double = {
    var result = w0
    otherW.activeIterator.foreach { case (index, value) =>
      val originalValue = this.w.getOrElse(index, 0.0)
      result += originalValue * value
    }
    result
  }

  /**
    * 转成字符串描述，用于saveModel等方法
    * @return 系数的字符串描述
    */
  override def toString: String = {
    //特征个数:系数个数
    val sb = new StringBuilder().append(this.size).append(":").append(w.size).append("\n")
    sb ++= w0.toString
    sb += '\n'
    //每个系数及其值
    w.foreach { case (index, value) =>
      sb ++= s"$index:$value"
      sb += '\n'
    }
    sb.deleteCharAt(sb.length - 1).mkString
  }

  /**
    * 计算系数的2范数
    * sum(abs(A).^p)^(1/p) where p=2
    *
    * @return 系数的2范数
    */
  override def norm(): Double = {
    math.sqrt(w.foldLeft(0.0) { case (sum: Double, (_, value: Double)) =>
      sum + value * value
    } + w0 * w0)
  }
}


object VectorCoefficients {
  /**
    * 根据字符串数组构造向量系数
    *
    * @param content 字符串数组
    * @return 向量系数
    */
  def apply(content: Array[String]): VectorCoefficients = {
    var codeArray = content(0).split(":")
    val size = codeArray(0).toInt
    val activeSize = codeArray(1).toInt
    val w = mutable.HashMap[Int, Double]()
    val w0 = content(1).toDouble

    for (i <- 2 to activeSize + 1) {
      codeArray = content(i).split(":")
      val index = codeArray(0).toInt
      val value = codeArray(1).toDouble
      w += index -> value
    }

    new VectorCoefficients(size, w0, w)
  }
}
