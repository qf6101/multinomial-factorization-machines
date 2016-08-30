package io.github.qf6101.mfm.tuning

import org.apache.spark.ml.param._

import scala.collection.mutable
import scala.util.Random

/**
  * User: qfeng
  * Date: 15-8-24 上午10:32
  * Usage: 快速模型选择时，参数的构建工具类
  */

class FastParamGridBuilder extends Serializable {
  val paramGrid = mutable.Map.empty[Param[_], Iterable[_]]

  def copy(): FastParamGridBuilder = {
    val result = new FastParamGridBuilder
    result.paramGrid ++= paramGrid
    result
  }

  /**
    * 对于参数集合，随机选择一组参数值
    *
    * @return 参数及对应参数值集合
    */
  def sampleParams(): ParamMap = {
    val paramMap = new ParamMap
    paramGrid.foreach { case (param, values) =>
      val valueList = values.toList
      val value = valueList(Random.nextInt(valueList.length))
      paramMap.put(param.asInstanceOf[Param[Any]], value)
    }
    paramMap
  }

  /**
    * Adds a double param with multiple values.
    */
  def addGrid(param: DoubleParam, values: Array[Double]): this.type = {
    addGrid[Double](param, values)
  }

  // specialized versions of addGrid for Java.

  /**
    * Adds a int param with multiple values.
    */
  def addGrid(param: IntParam, values: Array[Int]): this.type = {
    addGrid[Int](param, values)
  }

  /**
    * Adds a float param with multiple values.
    */
  def addGrid(param: FloatParam, values: Array[Float]): this.type = {
    addGrid[Float](param, values)
  }

  /**
    * Adds a param with multiple values (overwrites if the input param exists).
    */
  def addGrid[T](param: Param[T], values: Iterable[T]): this.type = {
    paramGrid.put(param, values)
    this
  }

  /**
    * Adds a long param with multiple values.
    */
  def addGrid(param: LongParam, values: Array[Long]): this.type = {
    addGrid[Long](param, values)
  }

  /**
    * Adds a boolean param with true and false.
    */
  def addGrid(param: BooleanParam): this.type = {
    addGrid[Boolean](param, Array(true, false))
  }

}
