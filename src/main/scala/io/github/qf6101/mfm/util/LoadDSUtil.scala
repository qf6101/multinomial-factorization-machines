package io.github.qf6101.mfm.util

import breeze.linalg.SparseVector
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
  * Created by qfeng on 15-3-18.
  */
object LoadDSUtil {
  // Convenient methods for `loadLibSVMFile`.

  /**
    * Loads labeled data in the LIBSVM format into an RDD[LabeledPoint], with the default number of
    * partitions.
    */
  def loadLibSVMDataSet(path: String,
                        numFeatures: Int = -1): (RDD[(Double, SparseVector[Double])], Int) = {
    val sc = SparkContext.getOrCreate()
    val dataSet = sc.textFile(path, sc.defaultMinPartitions)
    toLibSVMDataSet(dataSet, numFeatures)
  }

  /**
    * Loads labeled data in the LIBSVM format into an RDD[LabeledPoint].
    * The LIBSVM format is a text-based format used by LIBSVM and LIBLINEAR.
    * Each line represents a labeled sparse feature vector using the following format:
    * {{{label index1:value1 index2:value2 ...}}}
    * where the indices are one-based and in ascending order.
    * This method parses each line into a [[org.apache.spark.mllib.regression.LabeledPoint]],
    * where the feature indices are converted to zero-based.
    *
    * @param dataSet     数据集
    * @param numFeatures number of features, which will be determined from the input data if a
    *                    nonpositive value is given. This is useful when the dataset is already split
    *                    into multiple files and you want to load them separately, because some
    *                    features may not present in certain files, which leads to inconsistent
    *                    feature dimensions.
    * @return labeled data stored as an RDD[LabeledPoint]
    */
  def toLibSVMDataSet(dataSet: RDD[String],
                      numFeatures: Int = -1): (RDD[(Double, SparseVector[Double])], Int) = {
    val parsed = dataSet.map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
        val items = line.split(' ')
        val label = items.head.toDouble
        val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
          val indexAndValue = item.split(':')
          val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
        val value = indexAndValue(1).toDouble
          (index, value)
        }.unzip
        (label, indices, values)
      }

    // Determine number of features.
    val d = if (numFeatures > 0) {
      numFeatures
    } else {
      parsed.persist(StorageLevel.MEMORY_AND_DISK_SER)
      parsed.map { case (label, indices, values) =>
        indices.lastOption.getOrElse(0)
      }.reduce(math.max) + 1
    }

    (parsed.map { case (label, indices, values) =>
      (label, new SparseVector[Double](indices, values, d))
    }, d)
  }

}
