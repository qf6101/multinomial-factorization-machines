package io.github.qf6101.mfm.util

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext

/**
  * Created by qfeng on 16-3-3.
  */
object HDFSUtil {
  def deleteIfExists(file: String): Unit = {
    val sc = SparkContext.getOrCreate()
    val fs = FileSystem.get(sc.hadoopConfiguration)
    if (fs.exists(new Path(file))) {
      fs.delete(new Path(file), true)
    }
  }
}
