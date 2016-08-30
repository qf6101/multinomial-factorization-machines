package io.github.qf6101.mfm.base

/**
  * Created by qfeng on 15-3-12.
  */

/**
  * 模型系数，抽象基类
  */
abstract class Coefficients extends Serializable {
  /**
    * 只复制this的结构（比如参数个数），不复制内容
    *
    * @return 复制的拷贝
    */
  def copyEmpty(): Coefficients

  /**
    * 同时复制this的结构和内容
    *
    * @return 复制的拷贝
    */
  def copy: Coefficients

  /**
    * 对应系数加法，加至this上
    *
    * @param other 加数
    * @return this
    */
  def +=(other: Coefficients): Coefficients

  /**
    * 对应系数减法，减至this上
    *
    * @param other 减数
    * @return this
    */
  def -=(other: Coefficients): Coefficients

  /**
    *
    * 对应系数加法，加至复制this的类上
    *
    * @param other 加数
    * @return 加法结果（拷贝）
    */
  def +(other: Coefficients): Coefficients = {
    val result = this.copy
    result += other
    result
  }

  /**
    * 对应系数加上同一实数，加至复制this的类上
    *
    * @param addend 加数
    * @return 加法结果（拷贝）
    */
  def +(addend: Double): Coefficients

  /**
    * 对应系数减上同一实数，减至复制this的类上
    *
    * @param minuend 减数
    * @return 减法结果（拷贝）
    */
  def -(minuend: Double): Coefficients = {
    this.copy + (-minuend)
  }

  /**
    * 对应系数除上同一实数，加至复制this的类上
    *
    * @param dividend 除数
    * @return 除法结果
    */
  def /(dividend: Double): Coefficients

  /**
    * 对应系数乘上同一实数，加至复制this的类上
    *
    * @param multiplier  乘数
    * @return 乘法结果
    */
  def *(multiplier: Double): Coefficients

  /**
    * 转成字符串描述，用于saveModel等方法
    *
    * @return 系数的字符串描述
    */
  def toString(): String

  /**
    * 计算L2的正则值
    *
    * @param reg 正则参数
    * @return 参数加权后的L2正则值
    */
  def L2RegValue(reg: Array[Double]): Double

  /**
    * 计算L2的正则梯度值
    *
    * @param reg 正则参数
    * @return 参数加权后的L2正则梯度值
    */
  def L2RegGradient(reg: Array[Double]): Coefficients

  /**
    * 用L1稀疏化系数
    *
    * @param regParam 正则参数值
    * @param stepSize 学习率
    * @return 稀疏化后的系数
    */
  def L1Shrink(regParam: Array[Double], stepSize: Double): Coefficients

  /**
    * 计算L1的正则值
    *
    * @param regParam 正则参数
    * @return 参数绝对值加权后的L1正则值
    */
  def L1RegValue(regParam: Array[Double]): Double

  /**
    * 计算系数的2范数
    * sum(abs(A).^p)^(1/p) where p=2
    *
    * @return 系数的2范数
    */
  def norm: Double

  /**
    * 计算两组系数差异的2范数
    *
    * @param other 另一组系数
    * @return 差异的2范数值
    */
  def normDiff(other: Coefficients): Double = {
    (this - other).norm
  }

  /**
    * 对应系数减法，减至复制this的类上
    *
    * @param other 减数
    * @return 减法结果（拷贝）
    */
  def -(other: Coefficients): Coefficients = {
    val result = this.copy
    result -= other
    result
  }
}