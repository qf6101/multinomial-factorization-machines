package io.github.qf6101.mfm.util

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext

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
    val sc = SparkContext.getOrCreate()
    val fs = FileSystem.get(sc.hadoopConfiguration)
    if (fs.exists(new Path(file))) {
      fs.delete(new Path(file), true)
    }
  }
}
