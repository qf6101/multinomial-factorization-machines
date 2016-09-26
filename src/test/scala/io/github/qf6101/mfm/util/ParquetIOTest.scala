package io.github.qf6101.mfm.util

import org.apache.hadoop.fs.Path
import org.apache.parquet.hadoop.ParquetReader
import org.apache.parquet.tools.read.{SimpleReadSupport, SimpleRecord}

/**
  * Created by qfeng on 16-9-23.
  */
object ParquetIOTest {
  def main(args: Array[String]) {
    val reader = ParquetReader.builder[SimpleRecord](new SimpleReadSupport(),
      new Path("test_data/output/mnist/coefficient/coeff_data/0/coeff_data/w"))
      .build()
    var value = reader.read()
    while (value != null) {
      println(value.getValues.get(0).getValue.asInstanceOf[Double])
      value = reader.read()
    }
  }
}
