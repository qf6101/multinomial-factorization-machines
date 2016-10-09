package io.github.qf6101.mfm.util

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

/**
  * Created by qfeng on 16-3-3.
  */

/**
  * HDFS文件操作工具类
  */
object HDFSUtil {
  /**
    * 如果文件存在则删除它
    *
    * @param file 文件
    */
  def deleteIfExists(file: String): Unit = {
    val spark = SparkSession.builder().getOrCreate()
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    if (fs.exists(new Path(file))) {
      fs.delete(new Path(file), true)
    }
  }

  /**
    * 文件是否存在
    *
    * @param file 文件
    */
  def exists(file: String): Boolean = {
    val spark = SparkSession.builder().getOrCreate()
    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    fs.exists(new Path(file))
  }
}
