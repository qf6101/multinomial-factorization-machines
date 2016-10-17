package io.github.qf6101.mfm.optimization

/**
  * User: qfeng
  * Date: 15-12-29 下午4:49
  * Usage: SGD学习步长衰减类
  */
trait DecreasingStrategy extends Serializable {
  /**
    * 根据当前的迭代次数计算学习步长衰减的分母
    *
    * @param iter 迭代次数
    * @return 学习步长衰减的分母
    */
  def decrease(iter: Int): Double
}

class Log10DecreasingStrategy extends DecreasingStrategy {
  /**
    * 根据当前的迭代次数计算学习步长衰减的分母
    * 按照log10进行衰减，第91次迭代衰减为一半
    *
    * @param iter 迭代次数
    * @return 学习步长衰减的分母
    */
  def decrease(iter: Int): Double = {
    math.log10(9 + iter)
  }
}

class LogXDecreasingStrategy(X: Int) extends DecreasingStrategy {
  /**
    * 根据当前的迭代次数计算学习步长衰减的分母
    * 按照logX进行衰减
    *
    * @param iter 迭代次数
    * @return 学习步长衰减的分母
    */
  def decrease(iter: Int): Double = {
    math.log(X - 1 + iter) / math.log(X)
  }
}

class ConstantDecreasingStrategy(stepSize: Double) extends DecreasingStrategy {
  /**
    * 不衰减
    *
    * @param iter 迭代次数
    * @return 常数学习率(不衰减学习率)
    */
  def decrease(iter: Int): Double = {
    stepSize
  }
}

class sqrtDecreasingStrategy extends DecreasingStrategy {
  /**
    * 根据当前的迭代次数计算学习步长衰减的分母
    * 按照开方进行衰减，第5次迭代衰减为一半
    *
    * @param iter 迭代次数
    * @return 学习步长衰减的分母
    */
  def decrease(iter: Int): Double = {
    Math.sqrt(iter)
  }
}

