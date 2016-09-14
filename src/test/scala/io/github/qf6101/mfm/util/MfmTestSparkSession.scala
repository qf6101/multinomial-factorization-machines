package io.github.qf6101.mfm.util

import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, Suite}

/**
  * Created by qfeng on 15-3-13.
  */
trait MfmTestSparkSession extends BeforeAndAfterAll {
  self: Suite =>
  @transient var spark: SparkSession = _

  override def beforeAll() {
    super.beforeAll()
    spark = SparkSession.builder()
      .master("local[2]").appName(this.getClass.toString)
      .getOrCreate()
  }

  override def afterAll() {
    if (spark != null) {
      spark.stop()
    }
    super.afterAll()
  }
}